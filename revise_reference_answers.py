"""Audit and optionally revise golden answers using prior model outputs."""

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
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluation_results"
DEFAULT_SOURCES = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_AUDIT = DEFAULT_RESULTS / "reference_answer_revision_audit.json"
TARGET_MODELS = ("gpt-4o-mini", "gemini-3.5-flash", "veri")
MINIMUM_MODEL_RESULTS = 2


class ReferenceRevision(BaseModel):
    retrieval_context_answerable: bool = Field(
        description="Whether the supplied retrieval context answers the question"
    )
    corrected_expected_output: str = Field(
        description=(
            "A concise answer supported by the retrieval context, or a concise "
            "statement that the retrieval context does not specify the answer"
        )
    )
    retrieval_context_evidence: list[str] = Field(
        description="Short verbatim quotes from the retrieval context"
    )
    full_document_answerable: bool = Field(
        description="Whether the uploaded full document answers the question"
    )
    full_document_answer: str = Field(
        description="The answer from the full document, or an insufficient-information statement"
    )
    scope_mismatch: bool = Field(
        description=(
            "True only when answerability differs between retrieval context and full document"
        )
    )
    ambiguous_question: bool
    confidence: Literal["high", "medium", "low"]
    needs_human_review: bool
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the configured judge model to audit suspicious reference answers. "
            "API responses are checkpointed; --apply updates eligible golden fields."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--model",
        help="Audit model; defaults to judge.model from config.yaml.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override an auto-discovered result file; may be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum pending candidates to review; 0 reviews all.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--disagreements-only",
        action="store_true",
        help="Exclude unanimous NA/AN cases (not recommended; known bad labels may be missed).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply completed high-confidence, non-ambiguous revisions to the case file.",
    )
    parser.add_argument(
        "--reupload",
        action="store_true",
        help="Upload full source TXT files again instead of reusing audit file IDs.",
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


def load_judge_model(path: Path) -> str:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    try:
        model = config["judge"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain judge.model") from error
    if not isinstance(model, str) or not model.strip():
        raise ValueError("judge.model must be a non-empty string")
    return model.strip()


def parse_result_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if not separator or model not in TARGET_MODELS or not raw_path:
            raise ValueError(
                f"Invalid --result {value!r}; expected one of "
                f"{', '.join(TARGET_MODELS)}=PATH"
            )
        overrides[model] = Path(raw_path)
    return overrides


def discover_result_files(
    results_dir: Path, overrides: dict[str, Path]
) -> dict[str, Path]:
    discovered: dict[str, list[Path]] = {model: [] for model in TARGET_MODELS}
    for snapshot in results_dir.glob("*/config_snapshot.yaml"):
        try:
            with snapshot.open(encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        model = config.get("active_target_model") if isinstance(config, dict) else None
        results_file = snapshot.parent / "results.json"
        if model in discovered and results_file.is_file():
            discovered[model].append(results_file)

    selected = {}
    for model in TARGET_MODELS:
        if model in overrides:
            selected[model] = overrides[model]
        elif discovered[model]:
            selected[model] = sorted(discovered[model])[-1]
        else:
            raise ValueError(f"No results.json found for {model} in {results_dir}")
        if not selected[model].is_file():
            raise ValueError(f"Result file does not exist: {selected[model]}")
    return selected


def index_result_cases(result_files: dict[str, Path]) -> dict[str, dict[tuple[str, str], dict]]:
    indexes = {}
    for model, path in result_files.items():
        data = load_json(path)
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list):
            raise ValueError(f"{path} must contain a cases array")
        index = {}
        for case in cases:
            key = (case.get("document"), case.get("name"))
            if not all(isinstance(part, str) and part for part in key):
                raise ValueError(f"Invalid case identity in {path}")
            if key in index:
                raise ValueError(f"Duplicate case {key} in {path}")
            if case.get("status") == "completed":
                index[key] = case
        indexes[model] = index
    return indexes


def candidate_reasons(model_cases: dict[str, dict], disagreements_only: bool) -> list[str]:
    if len(model_cases) < MINIMUM_MODEL_RESULTS:
        return []
    states = {case.get("decision_state") for case in model_cases.values()}
    actuals = {case.get("actual_answered") for case in model_cases.values()}
    expected = next(iter(model_cases.values())).get("expected_answered")
    reasons = []
    if len(states) > 1:
        reasons.append("model_decision_disagreement")
    if not disagreements_only and expected is False and actuals == {True}:
        reasons.append("unanimous_NA")
    if not disagreements_only and expected is True and actuals == {False}:
        reasons.append("unanimous_AN")
    return reasons


def build_candidates(
    documents: list[dict],
    result_indexes: dict[str, dict[tuple[str, str], dict]],
    disagreements_only: bool,
) -> list[dict]:
    candidates = []
    for document in documents:
        document_name = document.get("name")
        for question in document.get("questions", []):
            key = (document_name, question.get("name"))
            model_cases = {
                model: index[key]
                for model, index in result_indexes.items()
                if key in index
            }
            reasons = candidate_reasons(model_cases, disagreements_only)
            if reasons:
                candidates.append(
                    {
                        "key": key,
                        "document": document,
                        "question": question,
                        "model_cases": model_cases,
                        "candidate_reasons": reasons,
                    }
                )
    return candidates


def index_source_files(source_dir: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for path in source_dir.glob("*.txt"):
        page_id = path.name.partition("-")[0]
        if page_id in sources:
            raise ValueError(f"Multiple source files found for page_id={page_id}")
        sources[page_id] = path
    return sources


def new_audit(
    cases_path: Path,
    result_files: dict[str, Path],
    judge_model: str,
    disagreements_only: bool,
) -> dict:
    return {
        "schema_version": 2,
        "cases_file": str(cases_path),
        "judge_model": judge_model,
        "selection": {
            "eligible_models": list(TARGET_MODELS),
            "minimum_model_results": MINIMUM_MODEL_RESULTS,
            "disagreements_only": disagreements_only,
        },
        "result_files": {model: str(path) for model, path in result_files.items()},
        "uploaded_files": {},
        "reviews": [],
    }


def load_or_create_audit(
    path: Path,
    cases_path: Path,
    result_files: dict[str, Path],
    judge_model: str,
    disagreements_only: bool,
) -> dict:
    if not path.exists():
        return new_audit(cases_path, result_files, judge_model, disagreements_only)
    audit = load_json(path)
    expected = new_audit(
        cases_path, result_files, judge_model, disagreements_only
    )
    legacy_selection = {
        "requires_all_models": list(TARGET_MODELS),
        "disagreements_only": disagreements_only,
    }
    if audit.get("schema_version") == 1 and audit.get("selection") == legacy_selection:
        audit["schema_version"] = expected["schema_version"]
        audit["selection"] = expected["selection"]
    for field in ("schema_version", "cases_file", "judge_model", "selection", "result_files"):
        if audit.get(field) != expected[field]:
            raise ValueError(
                f"Existing audit {path} has different {field}; use a different --audit path"
            )
    return audit


def upload_source(client: Any, path: Path) -> str:
    if path.stat().st_size >= 50 * 1024 * 1024:
        raise ValueError(f"Source file exceeds the 50 MB file-input limit: {path}")
    with path.open("rb") as file:
        uploaded = client.files.create(file=file, purpose="user_data")
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise ValueError(f"Upload returned no file ID for {path}")
    return file_id


def build_review_prompt(candidate: dict) -> str:
    document = candidate["document"]
    question = candidate["question"]
    context = "\n\n".join(document["retrieval_context"])
    answers = []
    for model in TARGET_MODELS:
        result = candidate["model_cases"].get(model)
        if result is None:
            answers.append(f"Model: {model}\nresult: unavailable")
        else:
            answers.append(
                f"Model: {model}\n"
                f"actual_answered: {result.get('actual_answered')}\n"
                f"decision_state: {result.get('decision_state')}\n"
                f"answer:\n{result.get('actual_output')}"
            )
    return (
        "Audit one RAG evaluation reference answer. The uploaded TXT is the full "
        "Wikipedia document. The retrieval context below is the exact, possibly "
        "truncated material shown to GPT and Gemini; it is the authoritative scope "
        "for expected_answered and corrected_expected_output. The full document is "
        "provided only to diagnose scope mismatch. Prior model answers are untrusted "
        "clues, not votes. Verify every claim against the texts. Do not preserve an "
        "incorrect reference merely because it agrees with a prior judge. Mark "
        "needs_human_review for ambiguity, conflicting evidence, or low confidence.\n\n"
        f"Document: {document.get('title')}\n"
        f"Question: {question.get('input')}\n"
        f"Current expected_answered: {question.get('expected_answered')}\n"
        f"Current expected_output: {question.get('expected_output')}\n\n"
        f"Exact retrieval context:\n---\n{context}\n---\n\n"
        "Prior model outputs:\n\n" + "\n\n".join(answers)
    )


def review_candidate(
    client: Any,
    model: str,
    file_id: str,
    candidate: dict,
) -> ReferenceRevision:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a conservative senior annotator auditing a RAG golden set. "
                    "Return only the requested structured assessment."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": build_review_prompt(candidate)},
                ],
            },
        ],
        text_format=ReferenceRevision,
    )
    if response.output_parsed is None:
        raise ValueError("Judge returned no parseable reference revision")
    revision = response.output_parsed
    if not revision.corrected_expected_output.strip():
        raise ValueError("Judge returned an empty corrected_expected_output")
    if revision.scope_mismatch != (
        revision.retrieval_context_answerable != revision.full_document_answerable
    ):
        raise ValueError("Judge returned inconsistent scope_mismatch")
    if revision.retrieval_context_answerable and not revision.retrieval_context_evidence:
        raise ValueError("Answerable revision must include retrieval-context evidence")
    return revision


def review_with_retries(
    client: Any,
    model: str,
    file_id: str,
    candidate: dict,
    retries: int,
) -> ReferenceRevision:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return review_candidate(client, model, file_id, candidate)
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


def review_identity(review: dict) -> tuple[str, str]:
    return review["document"], review["name"]


def make_review_record(candidate: dict, revision: ReferenceRevision) -> dict:
    question = candidate["question"]
    return {
        "status": "completed",
        "document": candidate["key"][0],
        "name": candidate["key"][1],
        "candidate_reasons": candidate["candidate_reasons"],
        "previous": {
            "expected_answered": question["expected_answered"],
            "expected_output": question["expected_output"],
        },
        "model_decisions": {
            model: {
                "actual_answered": result.get("actual_answered"),
                "decision_state": result.get("decision_state"),
            }
            for model, result in candidate["model_cases"].items()
        },
        "revision": revision.model_dump(),
        "applied": False,
    }


def apply_completed_reviews(documents: list[dict], audit: dict) -> Counter:
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
        eligible = (
            revision.get("confidence") == "high"
            and not revision.get("needs_human_review")
            and not revision.get("ambiguous_question")
        )
        if not eligible:
            counts["held_for_human_review"] += 1
            continue
        question = questions.get(review_identity(review))
        if question is None:
            raise ValueError(f"Reviewed question no longer exists: {review_identity(review)}")
        previous = review["previous"]
        current = {
            "expected_answered": question.get("expected_answered"),
            "expected_output": question.get("expected_output"),
        }
        proposed = {
            "expected_answered": revision["retrieval_context_answerable"],
            "expected_output": revision["corrected_expected_output"].strip(),
        }
        if current == proposed:
            review["applied"] = True
            counts["unchanged"] += 1
            continue
        if current != previous and not review.get("applied"):
            raise ValueError(
                f"Golden fields changed since review for {review_identity(review)}"
            )
        question.update(proposed)
        review["applied"] = True
        counts["updated"] += 1
    return counts


def summarize(audit: dict, candidates: list[dict]) -> dict:
    reviews = [review for review in audit["reviews"] if review.get("status") == "completed"]
    return {
        "candidate_count": len(candidates),
        "candidate_reasons": dict(
            Counter(reason for candidate in candidates for reason in candidate["candidate_reasons"])
        ),
        "completed_reviews": len(reviews),
        "failed_reviews": sum(review.get("status") == "failed" for review in audit["reviews"]),
        "proposed_answerability_changes": sum(
            review["previous"]["expected_answered"]
            != review["revision"]["retrieval_context_answerable"]
            for review in reviews
        ),
        "scope_mismatches": sum(review["revision"]["scope_mismatch"] for review in reviews),
        "needs_human_review": sum(
            review["revision"]["needs_human_review"]
            or review["revision"]["ambiguous_question"]
            or review["revision"]["confidence"] != "high"
            for review in reviews
        ),
        "applied": sum(bool(review.get("applied")) for review in reviews),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    try:
        load_env_file(args.env_file)
        judge_model = args.model.strip() if args.model else load_judge_model(args.config)
        if not judge_model:
            raise ValueError("--model must be a non-empty string")
        documents = load_json(args.cases)
        if not isinstance(documents, list):
            raise ValueError(f"{args.cases} must contain a JSON array")
        result_files = discover_result_files(
            args.results_dir, parse_result_overrides(args.result)
        )
        result_indexes = index_result_cases(result_files)
        candidates = build_candidates(
            documents, result_indexes, args.disagreements_only
        )
        sources = index_source_files(args.source_dir)
        audit = load_or_create_audit(
            args.audit,
            args.cases,
            result_files,
            judge_model,
            args.disagreements_only,
        )
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
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                f"OPENAI_API_KEY is not set in the environment or {args.env_file}"
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise SystemExit("Missing dependency; run: pip install openai") from error
        client = OpenAI()
        review_positions = {
            review_identity(review): index
            for index, review in enumerate(audit["reviews"])
        }
        reuploaded_documents = set()
        try:
            for position, candidate in enumerate(pending, 1):
                document = candidate["document"]
                page_id = str(document["page_id"])
                source = sources.get(page_id)
                if source is None:
                    raise ValueError(f"No full source TXT found for page_id={page_id}")
                uploaded = audit["uploaded_files"].get(document["name"])
                should_upload = not isinstance(uploaded, dict) or (
                    args.reupload and document["name"] not in reuploaded_documents
                )
                if should_upload:
                    file_id = upload_source(client, source)
                    audit["uploaded_files"][document["name"]] = {
                        "file_id": file_id,
                        "path": str(source),
                        "size": source.stat().st_size,
                    }
                    reuploaded_documents.add(document["name"])
                    save_json_atomic(audit, args.audit)
                else:
                    file_id = uploaded["file_id"]

                print(
                    f"[{position}/{len(pending)}] {document['title']} / "
                    f"{candidate['question']['name']}",
                    flush=True,
                )
                try:
                    revision = review_with_retries(
                        client,
                        judge_model,
                        file_id,
                        candidate,
                        args.retries,
                    )
                    record = make_review_record(candidate, revision)
                except RuntimeError as error:
                    record = {
                        "status": "failed",
                        "document": candidate["key"][0],
                        "name": candidate["key"][1],
                        "candidate_reasons": candidate["candidate_reasons"],
                        "error": str(error),
                        "applied": False,
                    }
                    print(f"  Failed: {error}", flush=True)

                key = candidate["key"]
                if key in review_positions:
                    audit["reviews"][review_positions[key]] = record
                else:
                    review_positions[key] = len(audit["reviews"])
                    audit["reviews"].append(record)
                audit["summary"] = summarize(audit, candidates)
                save_json_atomic(audit, args.audit)
        except KeyboardInterrupt:
            print("\nInterrupted. Every completed review is saved.", flush=True)
            raise SystemExit(130) from None
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error

    if args.apply:
        try:
            apply_counts = apply_completed_reviews(documents, audit)
            save_json_atomic(documents, args.cases)
            audit["summary"] = summarize(audit, candidates)
            audit["apply_summary"] = dict(apply_counts)
            save_json_atomic(audit, args.audit)
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error

    audit["summary"] = summarize(audit, candidates)
    save_json_atomic(audit, args.audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(f"Audit: {args.audit}")
    if args.apply:
        print(f"Updated cases: {args.cases}")
    else:
        print("No golden fields changed. Review the audit, then rerun with --apply.")


if __name__ == "__main__":
    main()