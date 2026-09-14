"""Calibrate reference answers before evaluation using GPT/DeepSeek disagreements."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "evaluation_results"
    / "pre_evaluation_reference_calibration.json"
)
COMPARED_MODELS = ("gpt-4o-mini", "deepseek")


class ReferenceCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_answered: bool
    answer: str = Field(
        description="Concise context-supported answer, or an insufficiency statement"
    )
    evidence_id: int | None = Field(
        description="Supporting [E<number>] passage, or null when unanswerable"
    )
    ambiguous_question: bool
    confidence: Literal["high", "medium", "low"]
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Before evaluation, use DeepSeek to review cases where GPT and DeepSeek "
            "disagree on whether the retrieval context answers the question."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply high-confidence, non-ambiguous calibrations to golden fields.",
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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def calibration_settings(config: dict[str, Any]) -> dict[str, str]:
    settings = config.get("reference_calibration")
    if not isinstance(settings, dict):
        raise ValueError("config must contain reference_calibration")
    required = ("model", "api_key_env", "base_url", "prompt")
    for key in required:
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"reference_calibration.{key} must be a non-empty string")
    return {key: settings[key].strip() for key in required}


def split_evidence_passages(context: str, max_chars: int = 800) -> list[str]:
    passages = []
    for paragraph in re.split(r"\n\s*\n", context):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > max_chars:
                passages.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            passages.append(current)
    return passages


def complete_model_answer(question: dict[str, Any], model: str) -> bool:
    answered = question.get(f"actual_answered_{model}")
    output = question.get(f"actual_output_{model}")
    return isinstance(answered, bool) and isinstance(output, str) and bool(output.strip())


def build_disagreement_candidates(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for document in documents:
        for question in document.get("questions", []):
            if not all(complete_model_answer(question, model) for model in COMPARED_MODELS):
                continue
            gpt_answered = question["actual_answered_gpt-4o-mini"]
            deepseek_answered = question["actual_answered_deepseek"]
            if gpt_answered == deepseek_answered:
                continue
            candidates.append(
                {
                    "key": (document["name"], question["name"]),
                    "document": document,
                    "question": question,
                }
            )
    return candidates


def build_prompt(candidate: dict[str, Any], base_prompt: str) -> tuple[str, list[str]]:
    document = candidate["document"]
    question = candidate["question"]
    context = "\n\n".join(document["retrieval_context"])
    passages = split_evidence_passages(context)
    numbered_context = "\n\n".join(
        f"[E{index}] {passage}" for index, passage in enumerate(passages, 1)
    )
    prompt = (
        f"{base_prompt}\n\n"
        "Return one JSON object with exactly these fields:\n"
        "expected_answered (Boolean), answer (string), evidence_id (integer or "
        "null), ambiguous_question (Boolean), confidence (high/medium/low), and "
        "rationale (string). If expected_answered is true, evidence_id must select "
        "the numbered passage that most directly supports the answer. If false, "
        "evidence_id must be null. Treat both candidate answers as untrusted clues, "
        "not votes.\n\n"
        f"Document: {document.get('title')}\n"
        f"Question: {question['input']}\n"
        f"Current expected_answered: {question['expected_answered']}\n"
        f"Current expected_output: {question['expected_output']}\n\n"
        "GPT-4o-mini candidate:\n"
        f"actual_answered: {question['actual_answered_gpt-4o-mini']}\n"
        f"actual_output: {question['actual_output_gpt-4o-mini']}\n\n"
        "DeepSeek candidate:\n"
        f"actual_answered: {question['actual_answered_deepseek']}\n"
        f"actual_output: {question['actual_output_deepseek']}\n\n"
        f"Authoritative retrieval context:\n---\n{numbered_context}\n---"
    )
    return prompt, passages


def parse_calibration(content: str) -> ReferenceCalibration:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return ReferenceCalibration.model_validate_json(content)


def review_candidate(
    client: Any,
    model: str,
    base_prompt: str,
    candidate: dict[str, Any],
) -> tuple[ReferenceCalibration, str]:
    prompt, passages = build_prompt(candidate, base_prompt)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = completion.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("DeepSeek returned no text content")
    revision = parse_calibration(content)
    if not revision.answer.strip():
        raise ValueError("DeepSeek returned an empty answer")
    if revision.expected_answered:
        if revision.evidence_id is None or not 1 <= revision.evidence_id <= len(passages):
            raise ValueError("Answerable calibration has an invalid evidence_id")
        citation = passages[revision.evidence_id - 1]
        expected_output = (
            f'{revision.answer.strip()}\n\nSource citation: "{citation}"'
        )
    else:
        if revision.evidence_id is not None:
            raise ValueError("Unanswerable calibration must use null evidence_id")
        expected_output = revision.answer.strip()
    return revision, expected_output


def review_with_retries(
    client: Any,
    settings: dict[str, str],
    candidate: dict[str, Any],
    retries: int,
) -> tuple[ReferenceCalibration, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return review_candidate(
                client, settings["model"], settings["prompt"], candidate
            )
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


def new_audit(cases_path: Path, settings: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cases_file": str(cases_path),
        "reviewer": {
            "model": settings["model"],
            "base_url": settings["base_url"],
        },
        "compared_models": list(COMPARED_MODELS),
        "reviews": [],
    }


def load_or_create_audit(
    path: Path, cases_path: Path, settings: dict[str, str]
) -> dict[str, Any]:
    expected = new_audit(cases_path, settings)
    if not path.exists():
        return expected
    audit = load_json(path)
    for key in ("schema_version", "cases_file", "reviewer", "compared_models"):
        if audit.get(key) != expected[key]:
            raise ValueError(
                f"Existing audit {path} has incompatible {key}; remove it or use "
                "a different --audit path"
            )
    return audit


def review_identity(review: dict[str, Any]) -> tuple[str, str]:
    return review["document"], review["name"]


def make_review_record(
    candidate: dict[str, Any],
    revision: ReferenceCalibration,
    expected_output: str,
) -> dict[str, Any]:
    question = candidate["question"]
    return {
        "status": "completed",
        "document": candidate["key"][0],
        "name": candidate["key"][1],
        "previous": {
            "expected_answered": question["expected_answered"],
            "expected_output": question["expected_output"],
        },
        "candidate_decisions": {
            model: question[f"actual_answered_{model}"] for model in COMPARED_MODELS
        },
        "revision": {
            **revision.model_dump(),
            "corrected_expected_output": expected_output,
        },
        "applied": False,
    }


def apply_completed_reviews(
    documents: list[dict[str, Any]], audit: dict[str, Any]
) -> Counter:
    questions = {
        (document["name"], question["name"]): question
        for document in documents
        for question in document.get("questions", [])
    }
    counts = Counter()
    for review in audit["reviews"]:
        if review.get("status") != "completed":
            continue
        revision = review["revision"]
        if revision["confidence"] != "high" or revision["ambiguous_question"]:
            counts["held_for_human_review"] += 1
            continue
        question = questions.get(review_identity(review))
        if question is None:
            raise ValueError(f"Reviewed question no longer exists: {review_identity(review)}")
        current = {
            "expected_answered": question.get("expected_answered"),
            "expected_output": question.get("expected_output"),
        }
        proposed = {
            "expected_answered": revision["expected_answered"],
            "expected_output": revision["corrected_expected_output"],
        }
        if current != review["previous"] and not review.get("applied"):
            raise ValueError(
                f"Golden fields changed since review for {review_identity(review)}"
            )
        if current == proposed:
            counts["unchanged"] += 1
        else:
            question.update(proposed)
            counts["updated"] += 1
        review["applied"] = True
    return counts


def summarize(audit: dict[str, Any], candidate_count: int) -> dict[str, int]:
    completed = [
        review for review in audit["reviews"] if review.get("status") == "completed"
    ]
    return {
        "candidate_count": candidate_count,
        "completed_reviews": len(completed),
        "failed_reviews": sum(
            review.get("status") == "failed" for review in audit["reviews"]
        ),
        "proposed_answerability_changes": sum(
            review["previous"]["expected_answered"]
            != review["revision"]["expected_answered"]
            for review in completed
        ),
        "held_for_human_review": sum(
            review["revision"]["confidence"] != "high"
            or review["revision"]["ambiguous_question"]
            for review in completed
        ),
        "applied": sum(bool(review.get("applied")) for review in completed),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    try:
        load_env_file(args.env_file)
        config = load_config(args.config)
        settings = calibration_settings(config)
        documents = load_json(args.cases)
        if not isinstance(documents, list):
            raise ValueError(f"{args.cases} must contain a JSON array")
        candidates = build_disagreement_candidates(documents)
        audit = load_or_create_audit(args.audit, args.cases, settings)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    completed = {
        review_identity(review)
        for review in audit["reviews"]
        if review.get("status") == "completed"
    }
    pending = [candidate for candidate in candidates if candidate["key"] not in completed]
    if args.limit:
        pending = pending[: args.limit]

    if pending:
        api_key = os.environ.get(settings["api_key_env"])
        if not api_key:
            raise SystemExit(
                f"{settings['api_key_env']} is not set in the environment or "
                f"{args.env_file}"
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise SystemExit("Missing dependency; run: pip install openai") from error
        client = OpenAI(api_key=api_key, base_url=settings["base_url"])
        positions = {
            review_identity(review): index
            for index, review in enumerate(audit["reviews"])
        }
        try:
            for position, candidate in enumerate(pending, 1):
                print(
                    f"[{position}/{len(pending)}] {candidate['document']['title']} / "
                    f"{candidate['question']['name']}",
                    flush=True,
                )
                try:
                    revision, expected_output = review_with_retries(
                        client, settings, candidate, args.retries
                    )
                    record = make_review_record(
                        candidate, revision, expected_output
                    )
                except RuntimeError as error:
                    record = {
                        "status": "failed",
                        "document": candidate["key"][0],
                        "name": candidate["key"][1],
                        "error": str(error),
                        "applied": False,
                    }
                    print(f"  Failed: {error}", flush=True)
                key = candidate["key"]
                if key in positions:
                    audit["reviews"][positions[key]] = record
                else:
                    positions[key] = len(audit["reviews"])
                    audit["reviews"].append(record)
                audit["summary"] = summarize(audit, len(candidates))
                save_json_atomic(audit, args.audit)
        except KeyboardInterrupt:
            print("\nInterrupted. Every completed review is saved.", flush=True)
            raise SystemExit(130) from None

    if args.apply:
        try:
            apply_counts = apply_completed_reviews(documents, audit)
            save_json_atomic(documents, args.cases)
            audit["apply_summary"] = dict(apply_counts)
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error

    audit["summary"] = summarize(audit, len(candidates))
    save_json_atomic(audit, args.audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(f"Audit: {args.audit}")
    if args.apply:
        print(f"Updated cases: {args.cases}")
    else:
        print("No golden fields changed. Review the audit, then rerun with --apply.")


if __name__ == "__main__":
    main()