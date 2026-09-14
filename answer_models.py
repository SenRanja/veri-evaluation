"""Answer cases concurrently with configured OpenAI-compatible candidate models."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actual_answered: bool = Field(
        description="Whether the context supports a substantive answer"
    )
    actual_output: str = Field(
        description="The answer with a verbatim source citation, or a refusal"
    )


@dataclass(frozen=True)
class CandidateModel:
    id: str
    model: str
    api_key_env: str
    base_url: str | None = None


@dataclass(frozen=True)
class AnswerTask:
    document_index: int
    question_index: int
    candidate: CandidateModel
    context: str
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer cases with all configured API candidate models. Network calls "
            "run concurrently; JSON updates are applied atomically by the main thread."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional candidate IDs to run; defaults to answering.models in config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum pending model-question tasks; 0 processes all.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate complete model-specific answer fields.",
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
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Case file must contain a JSON array: {path}")
    return data


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML object: {path}")
    return config


def load_candidates(
    config: dict[str, Any], selected_ids: list[str] | None = None
) -> list[CandidateModel]:
    answering = config.get("answering")
    if not isinstance(answering, dict):
        raise ValueError("config must contain answering")
    raw_models = answering.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("answering.models must be a non-empty list")

    candidates = []
    for index, raw in enumerate(raw_models, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"answering.models item {index} must be an object")
        candidate_id = raw.get("id")
        model = raw.get("model")
        api_key_env = raw.get("api_key_env")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (candidate_id, model, api_key_env)
        ):
            raise ValueError(
                f"answering.models item {index} requires id, model and api_key_env"
            )
        base_url = raw.get("base_url")
        if base_url is not None and (
            not isinstance(base_url, str) or not base_url.strip()
        ):
            raise ValueError(f"answering.models item {index} has invalid base_url")
        candidates.append(
            CandidateModel(
                id=candidate_id.strip(),
                model=model.strip(),
                api_key_env=api_key_env.strip(),
                base_url=base_url.strip().rstrip("/") if base_url else None,
            )
        )

    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("answering.models IDs must not contain duplicates")
    if selected_ids:
        unknown = set(selected_ids) - set(ids)
        if unknown:
            raise ValueError(f"Unknown candidate model IDs: {', '.join(sorted(unknown))}")
        selected = set(selected_ids)
        candidates = [candidate for candidate in candidates if candidate.id in selected]
    return candidates


def get_answering_settings(config: dict[str, Any]) -> tuple[str, int]:
    answering = config["answering"]
    prompt = answering.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("answering.prompt must be a non-empty string")
    max_workers = answering.get("max_workers", 4)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        raise ValueError("answering.max_workers must be a positive integer")
    return prompt.strip(), max_workers


def retrieval_context(document: dict[str, Any], document_index: int) -> str:
    contexts = document.get("retrieval_context")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(not isinstance(context, str) or not context.strip() for context in contexts)
    ):
        raise ValueError(f"Document {document_index} has no valid retrieval_context")
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


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def extract_source_citation(output: str) -> str | None:
    match = re.search(
        r"(?:^|\n)Source citation:\s*(?:\"([^\"]+)\"|(.+?))\s*$",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return (match.group(1) or match.group(2)).strip()


def validate_answer(answer: ModelAnswer, context: str) -> None:
    output = answer.actual_output.strip()
    if not output:
        raise ValueError("Model returned an empty actual_output")
    if not answer.actual_answered:
        return
    citation = extract_source_citation(output)
    if not citation:
        raise ValueError(
            'Answered output must end with Source citation: "<verbatim excerpt>"'
        )
    if normalize_text(citation) not in normalize_text(context):
        raise ValueError("Source citation is not a verbatim context excerpt")


def parse_model_answer(content: str) -> ModelAnswer:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return ModelAnswer.model_validate_json(content)


def ask_question(
    client: Any,
    candidate: CandidateModel,
    prompt: str,
    context: str,
    question: str,
    validation_feedback: str | None = None,
) -> ModelAnswer:
    feedback = (
        f"\nPrevious response validation error: {validation_feedback}\n"
        "Return a corrected response."
        if validation_feedback
        else ""
    )
    completion = client.chat.completions.create(
        model=candidate.model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}"
                    f"{feedback}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = completion.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Model returned no text content")
    answer = parse_model_answer(content)
    validate_answer(answer, context)
    return answer


def call_with_retries(
    client: Any,
    candidate: CandidateModel,
    prompt: str,
    context: str,
    question: str,
    retries: int,
) -> ModelAnswer:
    last_error: Exception | None = None
    validation_feedback = None
    for attempt in range(1, retries + 1):
        try:
            return ask_question(
                client,
                candidate,
                prompt,
                context,
                question,
                validation_feedback,
            )
        except Exception as error:
            last_error = error
            validation_feedback = str(error)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Maximum retries reached: {last_error}") from last_error


def build_tasks(
    documents: list[dict[str, Any]],
    candidates: list[CandidateModel],
    overwrite: bool,
    limit: int,
) -> tuple[list[AnswerTask], int]:
    tasks = []
    skipped = 0
    for document_index, document in enumerate(documents):
        context = retrieval_context(document, document_index + 1)
        for question_index, question in enumerate(document["questions"]):
            for candidate in candidates:
                answered_field = f"actual_answered_{candidate.id}"
                output_field = f"actual_output_{candidate.id}"
                complete = (
                    isinstance(question.get(answered_field), bool)
                    and isinstance(question.get(output_field), str)
                    and bool(question[output_field].strip())
                )
                if complete and not overwrite:
                    skipped += 1
                    continue
                if limit and len(tasks) >= limit:
                    return tasks, skipped
                tasks.append(
                    AnswerTask(
                        document_index=document_index,
                        question_index=question_index,
                        candidate=candidate,
                        context=context,
                        question=question["input"],
                    )
                )
    return tasks, skipped


def create_client(candidate: CandidateModel) -> Any:
    from openai import OpenAI

    api_key = os.environ.get(candidate.api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable {candidate.api_key_env}")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if candidate.base_url:
        kwargs["base_url"] = candidate.base_url
    return OpenAI(**kwargs)


def execute_tasks(
    documents: list[dict[str, Any]],
    tasks: list[AnswerTask],
    prompt: str,
    retries: int,
    max_workers: int,
    output_path: Path,
    client_factory: Any = create_client,
) -> tuple[int, list[str]]:
    thread_state = threading.local()

    def run_task(task: AnswerTask) -> ModelAnswer:
        clients = getattr(thread_state, "clients", None)
        if clients is None:
            clients = {}
            thread_state.clients = clients
        client = clients.get(task.candidate.id)
        if client is None:
            client = client_factory(task.candidate)
            clients[task.candidate.id] = client
        return call_with_retries(
            client,
            task.candidate,
            prompt,
            task.context,
            task.question,
            retries,
        )

    processed = 0
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[ModelAnswer], AnswerTask] = {
            executor.submit(run_task, task): task for task in tasks
        }
        try:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    answer = future.result()
                except Exception as error:
                    errors.append(
                        f"{task.candidate.id} document={task.document_index + 1} "
                        f"question={task.question_index + 1}: {error}"
                    )
                    continue

                question = documents[task.document_index]["questions"][
                    task.question_index
                ]
                question[f"actual_answered_{task.candidate.id}"] = (
                    answer.actual_answered
                )
                question[f"actual_output_{task.candidate.id}"] = (
                    answer.actual_output.strip()
                )
                save_json(documents, output_path)
                processed += 1
                print(
                    f"Saved {task.candidate.id} document={task.document_index + 1} "
                    f"question={task.question_index + 1}",
                    flush=True,
                )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    return processed, errors


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    try:
        load_env_file(args.env_file)
        config = load_config(args.config)
        candidates = load_candidates(config, args.models)
        prompt, max_workers = get_answering_settings(config)
        documents = load_json(args.input)
        validate_documents(documents)
        for candidate in candidates:
            if not os.environ.get(candidate.api_key_env):
                raise ValueError(
                    f"Missing environment variable {candidate.api_key_env}"
                )
        tasks, skipped = build_tasks(
            documents, candidates, args.overwrite, args.limit
        )
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(
        f"Models: {', '.join(candidate.id for candidate in candidates)}; "
        f"pending tasks: {len(tasks)}; workers: {max_workers}",
        flush=True,
    )
    try:
        processed, errors = execute_tasks(
            documents,
            tasks,
            prompt,
            args.retries,
            max_workers,
            args.input,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Every response reported as saved is in the JSON file.")
        raise SystemExit(130) from None

    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(errors)}")
    for error in errors:
        print(f"  {error}")
    print(f"Updated file: {args.input}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()