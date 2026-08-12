#!/usr/bin/env python3
"""Merge evaluation case files without discarding model-specific fields."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_GPT_INPUT = Path("evaluation_cases/test_cases_novel_withGPTmini4o.json")
DEFAULT_VERI_INPUT = Path("evaluation_cases/test_cases_novel.json")
DEFAULT_OUTPUT = Path("evaluation_cases/test_cases_novel.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge documents and questions from two evaluation case JSON files."
    )
    parser.add_argument("--first", type=Path, default=DEFAULT_GPT_INPUT)
    parser.add_argument("--second", type=Path, default=DEFAULT_VERI_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    if not isinstance(documents, list):
        raise ValueError(f"{path} must contain a JSON array")
    return documents


def merge_fields(
    first: dict[str, Any],
    second: dict[str, Any],
    location: str,
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded or set()
    merged = {key: value for key, value in first.items() if key not in excluded}
    for key, value in second.items():
        if key in excluded:
            continue
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting value at {location}.{key}")
        merged[key] = value
    return merged


def index_unique(
    items: list[dict[str, Any]], key: str, location: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(items, 1):
        if not isinstance(item, dict) or key not in item:
            raise ValueError(f"{location}[{position}] must contain {key}")
        identity = str(item[key])
        if identity in indexed:
            raise ValueError(f"Duplicate {key}={identity} in {location}")
        indexed[identity] = item
    return indexed


def merge_questions(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    page_id: str,
) -> list[dict[str, Any]]:
    first_by_name = index_unique(first, "name", f"page_id={page_id}.questions")
    second_by_name = index_unique(second, "name", f"page_id={page_id}.questions")
    names = list(first_by_name) + [
        name for name in second_by_name if name not in first_by_name
    ]
    return [
        merge_fields(
            first_by_name.get(name, {}),
            second_by_name.get(name, {}),
            f"page_id={page_id}.questions.{name}",
        )
        for name in names
    ]


def merge_documents(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    first_by_page = index_unique(first, "page_id", "first")
    second_by_page = index_unique(second, "page_id", "second")
    page_ids = list(first_by_page) + [
        page_id for page_id in second_by_page if page_id not in first_by_page
    ]
    merged_documents = []
    for page_id in page_ids:
        first_document = first_by_page.get(page_id, {})
        second_document = second_by_page.get(page_id, {})
        merged = merge_fields(
            first_document,
            second_document,
            f"page_id={page_id}",
            excluded={"questions"},
        )
        merged["questions"] = merge_questions(
            first_document.get("questions", []),
            second_document.get("questions", []),
            page_id,
        )
        merged_documents.append(merged)
    return merged_documents


def save_json_atomic(documents: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(documents, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    first = load_documents(args.first)
    second = load_documents(args.second)
    merged = merge_documents(first, second)
    save_json_atomic(merged, args.output)
    print(f"Merged documents: {len(merged)}")
    print(f"Merged questions: {sum(len(item['questions']) for item in merged)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()