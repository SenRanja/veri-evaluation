"""Upload Wikipedia case files to Veris AI and checkpoint its answers."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_FILES_DIRECTORY = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
FILES_ENDPOINT = "https://verisai.duckdns.org/api/v1/files/"
CHAT_ENDPOINT = "https://verisai.duckdns.org/api/chat/completions"
MODEL = "arena-model"

REFUSAL_MARKERS = (
    "\u65e0\u76f8\u5173 segment",
    "\u65e0\u76f8\u5173\u6bb5\u843d",
    "\u7121\u76f8\u95dc\u6bb5\u843d",
    "\u771f\u610f\u67e5\u8be2\u5931\u8d25",
    "\u672a\u641c\u7d22\u5230\u77e5\u8bc6\u5e93\u4e2d\u7684\u76f8\u5173\u4fe1\u606f",
    "\u672a\u641c\u5c0b\u5230\u77e5\u8b58\u5eab\u4e2d\u7684\u76f8\u95dc\u8cc7\u8a0a",
    "\u6ca1\u6709\u5217\u51fa",
    "\u6c92\u6709\u5217\u51fa",
    "\u672a\u63d0\u53ca",
    "\u6ca1\u6709\u63d0\u53ca",
    "\u6c92\u6709\u63d0\u53ca",
    "\u5e76\u6ca1\u6709\u63d0\u5230",
    "\u4e26\u6c92\u6709\u63d0\u5230",
    "\u6ca1\u6709\u63d0\u5230",
    "\u6c92\u6709\u63d0\u5230",
    "\u6ca1\u6709\u63d0\u4f9b",
    "\u6c92\u6709\u63d0\u4f9b",
    "no relevant segment",
    "no relevant passage",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "does not specify",
    "doesn't specify",
    "does not list",
    "doesn't list",
    "does not mention",
    "doesn't mention",
    "not specified in",
    "not mentioned in",
    "insufficient information",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload case TXT files to Veris AI, ask each question in order, "
            "and checkpoint the results in the case JSON."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--files-directory", type=Path, default=DEFAULT_FILES_DIRECTORY)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--start-document",
        type=int,
        default=1,
        help="One-based document position at which to start (default: 1).",
    )
    parser.add_argument(
        "--document-limit",
        type=int,
        default=0,
        help="Process only the first N documents; 0 processes all documents.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="API attempts per upload or question (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing Veris answers while reusing uploaded files.",
    )
    parser.add_argument(
        "--reupload",
        action="store_true",
        help="Upload source files again even when veri_file_id already exists.",
    )
    parser.add_argument(
        "--print-api-response",
        action="store_true",
        help="Print the complete HTTP status, response headers, and body for each chat call.",
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


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def index_source_files(directory: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for source_file in directory.glob("*.txt"):
        page_id, separator, _ = source_file.name.partition("-")
        if separator:
            indexed.setdefault(page_id, []).append(source_file)
    return indexed


def find_source_file(
    document: dict[str, Any], source_files: dict[str, list[Path]], directory: Path
) -> Path:
    page_id = str(document["page_id"])
    matches = sorted(source_files.get(page_id, []))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one TXT for page_id {page_id}, "
            f"found {len(matches)} in {directory}"
        )
    return matches[0]


def extract_response_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        raise ValueError("Veris returned an unsupported response body")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

    for key in ("content", "response", "answer", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
    raise ValueError("Veris response did not contain answer text")


def build_question_prompt(question: str, title: str, filename: str) -> str:
    return (
        f"{question}\n\n"
        "Please answer using only the attached file. End the response with the "
        "headings 【Answer】, 【Cited passage】, and 【Source】 in that order. "
        "After 【Answer】 write the answer. After 【Cited passage】 quote the "
        "supporting passage exactly as it appears in the file. After 【Source】 "
        f"write: {title} (filename: {filename}). Do not repeat these instructions "
        "or write placeholder text. If the file has no relevant passage, state "
        "that the question cannot be answered and write No relevant passage after "
        "【Cited passage】."
    )


def response_indicates_refusal(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def extract_answer_section(answer: str) -> str:
    if "【Answer】" not in answer or "【Cited passage】" not in answer:
        return answer.strip()
    return answer.split("【Answer】", 1)[1].split("【Cited passage】", 1)[0].strip()


def normalize_citation_output(answer: str, title: str, filename: str) -> str:
    required_sections = ("【Answer】", "【Cited passage】", "【Source】")
    if all(section in answer for section in required_sections):
        return answer
    cited_passage = (
        "No relevant passage"
        if response_indicates_refusal(answer)
        else "Not returned separately by the Veris API"
    )
    return (
        f"【Answer】\n{answer.strip()}\n\n"
        f"【Cited passage】\n{cited_passage}\n\n"
        f"【Source】\n{title} (filename: {filename})"
    )


def add_source_index(answer: str, file_id: str, title: str, filename: str) -> str:
    if "【Source index】" in answer:
        return answer
    return (
        f"{answer.rstrip()}\n\n"
        "【Source index】\n"
        f"file_id: {file_id}\n"
        f"filename: {filename}\n"
        f"title: {title}"
    )


def print_api_response(response: requests.Response) -> None:
    print("  --- Complete Veris API response ---")
    print(f"  HTTP {response.status_code}")
    print(json.dumps(dict(response.headers), ensure_ascii=False, indent=2))
    print(response.text)
    print("  --- End Veris API response ---")


def upload_file(
    session: requests.Session,
    api_key: str,
    source_file: Path,
    timeout: float,
) -> str:
    with source_file.open("rb") as file:
        response = session.post(
            FILES_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (source_file.name, file, "text/plain")},
            data={"process": "true"},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    file_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(file_id, str) or not file_id.strip():
        raise ValueError("Veris upload response did not contain a file id")
    return file_id.strip()


def ask_question(
    session: requests.Session,
    api_key: str,
    file_id: str,
    question: str,
    title: str,
    filename: str,
    timeout: float,
    show_api_response: bool,
) -> str:
    response = session.post(
        CHAT_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": build_question_prompt(question, title, filename),
                }
            ],
            "features": {"true_meaning_search": True},
            "files": [{"id": file_id, "type": "file"}],
            "chat_id": str(uuid.uuid4()),
            "id": str(uuid.uuid4()),
            "stream": False,
        },
        timeout=timeout,
    )
    if show_api_response:
        print_api_response(response)
    response.raise_for_status()
    answer = extract_response_text(response.json())
    if not answer:
        raise ValueError("Veris returned an empty answer")
    return normalize_citation_output(answer, title, filename)


def call_with_retries(operation: Any, retries: int, description: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"  {description} attempt {attempt} failed: {error}; "
                    f"retrying in {wait_seconds} second(s)."
                )
                time.sleep(wait_seconds)
    raise RuntimeError(f"{description} failed after {retries} attempts: {last_error}")


def is_answered(answer: str) -> bool:
    return not response_indicates_refusal(extract_answer_section(answer))


def main() -> None:
    args = parse_args()
    if args.start_document < 1:
        raise SystemExit("--start-document must be at least 1")
    if args.document_limit < 0:
        raise SystemExit("--document-limit cannot be negative")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")

    try:
        load_env_file(args.env_file)
        documents = load_json(args.input)
        start_index = args.start_document - 1
        end_index = start_index + args.document_limit if args.document_limit else None
        selected = documents[start_index:end_index]
        source_files = index_source_files(args.files_directory)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    api_key = os.environ.get("VERI_API_KEY")
    if not api_key:
        raise SystemExit(f"VERI_API_KEY is not set in the environment or {args.env_file}")

    uploaded = 0
    answered = 0
    skipped = 0
    failed_documents = 0
    failed_questions = 0
    session = requests.Session()

    try:
        selection_end = args.start_document + len(selected) - 1
        for document_index, document in enumerate(selected, args.start_document):
            try:
                source_file = find_source_file(
                    document, source_files, args.files_directory
                )
                file_id = document.get("veri_file_id")
                if args.reupload or not isinstance(file_id, str) or not file_id.strip():
                    print(
                        f"[{document_index}/{selection_end}] Uploading "
                        f"{source_file.name}"
                    )
                    file_id = call_with_retries(
                        lambda: upload_file(session, api_key, source_file, args.timeout),
                        args.retries,
                        "Upload",
                    )
                    document["veri_file_id"] = file_id
                    save_json(documents, args.input)
                    uploaded += 1
                    print("  Saved veri_file_id")
                else:
                    file_id = file_id.strip()
                questions = document["questions"]
                title = document["title"]
            except Exception as error:
                failed_documents += 1
                print(
                    f"[{document_index}/{selection_end}] Skipped document: {error}"
                )
                continue

            for question_index, question in enumerate(questions, 1):
                try:
                    complete = (
                        isinstance(question.get("actual_answered_veri"), bool)
                        and isinstance(question.get("actual_output_veri"), str)
                        and bool(question["actual_output_veri"].strip())
                    )
                    has_citation = all(
                        section in question.get("actual_output_veri", "")
                        for section in ("【Answer】", "【Cited passage】", "【Source】")
                    )
                    if complete and has_citation and not args.overwrite:
                        indexed_output = add_source_index(
                            question["actual_output_veri"],
                            file_id,
                            title,
                            source_file.name,
                        )
                        if question["actual_output_veri"] != indexed_output:
                            question["actual_output_veri"] = indexed_output
                            save_json(documents, args.input)
                            print(
                                f"[{document_index}/{selection_end}] Indexed "
                                f"{question.get('name', f'question_{question_index}')} "
                                "actual_output_veri"
                            )
                        skipped += 1
                        continue

                    name = question.get("name", f"question_{question_index}")
                    print(f"[{document_index}/{selection_end}] {title} / {name}")
                    answer = call_with_retries(
                        lambda: ask_question(
                            session,
                            api_key,
                            file_id,
                            question["input"],
                            title,
                            source_file.name,
                            args.timeout,
                            args.print_api_response,
                        ),
                        args.retries,
                        "Question",
                    )
                    answer = add_source_index(
                        answer,
                        file_id,
                        title,
                        source_file.name,
                    )
                    question["actual_answered_veri"] = is_answered(answer)
                    question["actual_output_veri"] = answer
                    save_json(documents, args.input)
                    answered += 1
                    print(
                        "  Saved actual_answered_veri="
                        f"{question['actual_answered_veri']} and actual_output_veri"
                    )
                except Exception as error:
                    failed_questions += 1
                    print(
                        f"[{document_index}/{selection_end}] Skipped "
                        f"question_{question_index}: {error}"
                    )
                    continue
    except KeyboardInterrupt:
        print("\nInterrupted. Every item reported as saved is in the JSON file.")
        raise SystemExit(130) from None
    finally:
        session.close()

    print(f"Uploaded files: {uploaded}")
    print(f"Answered questions: {answered}")
    print(f"Skipped existing questions: {skipped}")
    print(f"Failed documents skipped: {failed_documents}")
    print(f"Failed questions skipped: {failed_questions}")
    print(f"Updated file: {args.input}")


if __name__ == "__main__":
    main()