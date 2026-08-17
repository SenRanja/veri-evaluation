"""Use the configured judge model to classify whether Veris answered each question."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class AnswerDecision(BaseModel):
    actual_answered: bool = Field(
        description=(
            "True when the response attempts a substantive answer; false when it "
            "refuses or says the requested information was not found or specified"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-fill actual_answered_veri using the configured judge model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum outputs to classify; 0 classifies all remaining outputs.",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="API attempts per output (default: 3)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reclassify outputs already marked as judged by the configured model.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env entry at {path}:{line_number}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            os.environ.setdefault(key, value)


def load_inputs(config_path: Path, input_path: Path) -> tuple[str, list[dict[str, Any]]]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    try:
        judge_model = config["judge"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain judge.model") from error
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("judge.model must be a non-empty string")

    with input_path.open(encoding="utf-8") as file:
        documents = json.load(file)
    if not isinstance(documents, list):
        raise ValueError(f"Case file must contain a JSON array: {input_path}")
    return judge_model.strip(), documents


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def classify_response(client: Any, model: str, question: str, response: str) -> bool:
    result = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Classify whether an assistant response actually answers the user's "
                    "question. Set actual_answered=true when it gives or attempts a "
                    "substantive answer, even if the answer may be incorrect. Set it to "
                    "false when it refuses, says information is unavailable/not found/not "
                    "specified, or contains only citation/source boilerplate. Judge the "
                    "Answer section when labeled sections are present. Do not assess factual "
                    "correctness and do not answer the question yourself."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nAssistant response:\n{response}",
            },
        ],
        text_format=AnswerDecision,
    )
    if result.output_parsed is None:
        raise ValueError("Judge returned no parseable decision")
    return result.output_parsed.actual_answered


def classify_with_retries(
    client: Any, model: str, question: str, response: str, retries: int
) -> bool:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return classify_response(client, model, question, response)
        except Exception as error:
            last_error = error
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"  Attempt {attempt} failed: {error}; retrying in "
                    f"{wait_seconds} second(s).",
                    flush=True,
                )
                time.sleep(wait_seconds)
    raise RuntimeError(f"Maximum retries reached: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    try:
        load_env_file(args.env_file)
        judge_model, documents = load_inputs(args.config, args.input)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(f"OPENAI_API_KEY is not set in the environment or {args.env_file}")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Missing dependency; run: pip install openai") from error

    client = OpenAI()
    processed = 0
    skipped = 0
    failed = 0
    total_documents = len(documents)
    try:
        for document_index, document in enumerate(documents, 1):
            questions = document.get("questions", [])
            for question_index, question in enumerate(questions, 1):
                if args.limit and processed >= args.limit:
                    break
                if (
                    question.get("actual_answered_veri_judged_by") == judge_model
                    and not args.overwrite
                ):
                    skipped += 1
                    continue
                output = question.get("actual_output_veri")
                user_question = question.get("input")
                if not isinstance(output, str) or not output.strip():
                    failed += 1
                    print(
                        f"[{document_index}/{total_documents}] Skipped question "
                        f"{question_index}: missing actual_output_veri",
                        flush=True,
                    )
                    continue
                if not isinstance(user_question, str) or not user_question.strip():
                    failed += 1
                    print(
                        f"[{document_index}/{total_documents}] Skipped question "
                        f"{question_index}: missing input",
                        flush=True,
                    )
                    continue

                name = question.get("name", f"question_{question_index}")
                print(f"[{document_index}/{total_documents}] {name}", flush=True)
                try:
                    decision = classify_with_retries(
                        client,
                        judge_model,
                        user_question,
                        output,
                        args.retries,
                    )
                except RuntimeError as error:
                    failed += 1
                    print(f"  Skipped after retries: {error}", flush=True)
                    continue
                question["actual_answered_veri"] = decision
                question["actual_answered_veri_judged_by"] = judge_model
                save_json(documents, args.input)
                processed += 1
                print(f"  Saved actual_answered_veri={decision}", flush=True)
            if args.limit and processed >= args.limit:
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Every reported decision is saved.", flush=True)
        raise SystemExit(130) from None

    print(f"Judge model: {judge_model}")
    print(f"Processed: {processed}")
    print(f"Skipped judged: {skipped}")
    print(f"Failed/skipped: {failed}")
    print(f"Updated file: {args.input}")


if __name__ == "__main__":
    main()