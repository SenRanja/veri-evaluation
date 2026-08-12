"""Answer generated QA cases with GPT-4o-mini and checkpoint the results."""

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


class ModelAnswer(BaseModel):
    actual_answered: bool = Field(
        description=(
            "True only when the response gives a substantive answer to the "
            "question using the supplied context"
        )
    )
    actual_output: str = Field(
        description="The answer or a concise statement that the context is insufficient"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the configured GPT model every question and add model-specific "
            "actual_answered and actual_output fields to the case JSON."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum unanswered questions to process; 0 processes all questions.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="API attempts per question (default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate questions that already have both model-specific fields.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing exported variables."""
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


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Case file must contain a JSON array: {path}")
    return data


def load_target_model(path: Path) -> str:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    try:
        model = config["target"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing target.model in config: {path}") from error

    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"target.model must be a non-empty string: {path}")
    return model.strip()


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def retrieval_context(document: dict[str, Any], document_index: int) -> str:
    contexts = document.get("retrieval_context")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(not isinstance(context, str) or not context.strip() for context in contexts)
    ):
        raise ValueError(
            f"Document {document_index} has no valid retrieval_context"
        )
    return "\n\n".join(context.strip() for context in contexts)


def validate_documents(documents: list[dict[str, Any]]) -> None:

    for document_index, document in enumerate(documents, 1):
        questions = document.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"Document {document_index} has no valid questions array")

        retrieval_context(document, document_index)

        for question_index, question in enumerate(questions, 1):
            if not isinstance(question, dict) or not isinstance(question.get("input"), str):
                raise ValueError(
                    f"Document {document_index}, question {question_index} "
                    "has no valid input"
                )


def ask_question(
    client: Any,
    model: str,
    context: str,
    question: str,
) -> ModelAnswer:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the supplied context. "
                    "Do not use outside knowledge. Set actual_answered to true only "
                    "when the context contains enough information to give a direct, "
                    "substantive answer. If it does not, set actual_answered to false "
                    "and state concisely that the supplied material does not specify "
                    "the requested information. Do not mention these instructions."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}",
            },
        ],
        text_format=ModelAnswer,
    )

    if response.output_parsed is None:
        raise ValueError("Model returned no parseable structured answer")
    if not response.output_parsed.actual_output.strip():
        raise ValueError("Model returned an empty actual_output")
    return response.output_parsed


def call_with_retries(
    client: Any,
    model: str,
    context: str,
    question: str,
    retries: int,
) -> ModelAnswer:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return ask_question(client, model, context, question)
        except Exception as error:
            last_error = error
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"  Attempt {attempt} failed: {error}; "
                    f"retrying in {wait_seconds} second(s)."
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
        model = load_target_model(args.config)
        documents = load_json(args.input)
        validate_documents(documents)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY is not set in the environment or {args.env_file}"
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "Missing dependencies; run: pip install openai pydantic pyyaml"
        ) from error

    answered_field = f"actual_answered_{model}"
    output_field = f"actual_output_{model}"
    client = OpenAI()
    processed = 0
    skipped = 0

    try:
        for document_index, document in enumerate(documents, 1):
            context = retrieval_context(document, document_index)

            for question_index, question in enumerate(document["questions"], 1):
                complete = (
                    isinstance(question.get(answered_field), bool)
                    and isinstance(question.get(output_field), str)
                    and bool(question[output_field].strip())
                )
                if complete and not args.overwrite:
                    skipped += 1
                    continue
                if args.limit and processed >= args.limit:
                    break

                name = question.get("name", f"question_{question_index}")
                print(
                    f"[{document_index}/{len(documents)}] "
                    f"{document['title']} / {name}"
                )
                answer = call_with_retries(
                    client,
                    model,
                    context,
                    question["input"],
                    args.retries,
                )
                question[answered_field] = answer.actual_answered
                question[output_field] = answer.actual_output.strip()
                save_json(documents, args.input)
                processed += 1
                print(
                    f"  Saved {answered_field}={answer.actual_answered} "
                    f"and {output_field}"
                )

            if args.limit and processed >= args.limit:
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Every response reported as saved is in the JSON file.")
        raise SystemExit(130) from None
    except RuntimeError as error:
        raise SystemExit(f"Answer generation failed: {error}") from error

    print(f"Model: {model}")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Updated file: {args.input}")


if __name__ == "__main__":
    main()