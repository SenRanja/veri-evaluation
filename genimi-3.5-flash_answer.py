"""Answer QA cases with Gemini 3.5 Flash and checkpoint each result."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "gemini-3.5-flash"


class ModelAnswer(BaseModel):
    actual_answered: bool = Field(
        description=(
            "True only when the response gives a substantive answer to the "
            "question using the supplied context"
        )
    )
    actual_output: str = Field(
        description="The answer or a concise statement that context is insufficient"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask Gemini 3.5 Flash every question and atomically save "
            "model-specific answer fields."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
        help="Regenerate questions that already have both Gemini fields.",
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


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)
    if not isinstance(documents, list):
        raise ValueError(f"Case file must contain a JSON array: {path}")
    return documents


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def get_retrieval_context(document: dict[str, Any]) -> str:
    contexts = document.get("retrieval_context")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(not isinstance(context, str) or not context.strip() for context in contexts)
    ):
        raise ValueError("missing valid retrieval_context")
    return "\n\n".join(context.strip() for context in contexts)


def ask_question(client: Any, model: str, context: str, question: str) -> ModelAnswer:
    prompt = (
        "Answer the question using only the supplied context. Do not use outside "
        "knowledge. Set actual_answered to true only when the context contains "
        "enough information for a direct, substantive answer. Otherwise set it "
        "to false and state concisely that the supplied material does not specify "
        "the requested information. Do not mention these instructions.\n\n"
        f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}"
    )
    response = client.interactions.create(
        model=model,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ModelAnswer.model_json_schema(),
        },
    )
    answer = ModelAnswer.model_validate_json(response.output_text)
    if not answer.actual_output.strip():
        raise ValueError("Gemini returned an empty actual_output")
    return answer


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
    if not isinstance(args.model, str) or not args.model.strip():
        raise SystemExit("--model must be a non-empty string")

    try:
        load_env_file(args.env_file)
        documents = load_json(args.input)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY is not set or missing from {args.env_file}")
    try:
        from google import genai
    except ImportError as error:
        raise SystemExit("Missing dependency; run: pip install google-genai") from error

    model = args.model.strip()
    answered_field = f"actual_answered_{model}"
    output_field = f"actual_output_{model}"
    client = genai.Client(api_key=api_key)
    processed = 0
    skipped_existing = 0
    skipped_errors = 0
    total_documents = len(documents)

    try:
        for document_index, document in enumerate(documents, 1):
            if args.limit and processed >= args.limit:
                break
            try:
                context = get_retrieval_context(document)
                questions = document["questions"]
                if not isinstance(questions, list):
                    raise ValueError("missing questions array")
                title = document.get("title", f"document_{document_index}")
            except (KeyError, TypeError, ValueError) as error:
                skipped_errors += 1
                print(
                    f"[{document_index}/{total_documents}] Skipped document: {error}",
                    flush=True,
                )
                continue

            for question_index, question in enumerate(questions, 1):
                complete = (
                    isinstance(question.get(answered_field), bool)
                    and isinstance(question.get(output_field), str)
                    and bool(question[output_field].strip())
                )
                if complete and not args.overwrite:
                    skipped_existing += 1
                    continue
                if args.limit and processed >= args.limit:
                    break

                name = question.get("name", f"question_{question_index}")
                print(
                    f"[{document_index}/{total_documents}] {title} / {name}",
                    flush=True,
                )
                try:
                    question_text = question["input"]
                    if not isinstance(question_text, str) or not question_text.strip():
                        raise ValueError("missing question input")
                    answer = call_with_retries(
                        client,
                        model,
                        context,
                        question_text,
                        args.retries,
                    )
                except (KeyError, RuntimeError, ValueError) as error:
                    skipped_errors += 1
                    print(f"  Skipped question: {error}", flush=True)
                    continue

                question[answered_field] = answer.actual_answered
                question[output_field] = answer.actual_output.strip()
                save_json(documents, args.input)
                processed += 1
                print(
                    f"  Saved {answered_field}={answer.actual_answered} and "
                    f"{output_field}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted. Every reported answer is saved.", flush=True)
        raise SystemExit(130) from None
    finally:
        client.close()

    print(f"Model: {model}")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped errors: {skipped_errors}")
    print(f"Updated file: {args.input}")


if __name__ == "__main__":
    main()