"""Extract the text field from a Wikipedia JSONL dataset into text files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path("./evaluation_cases/wikipedia_10000.jsonl")
DEFAULT_OUTPUT = Path("./evaluation_cases/test_cases_novel")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Wikipedia JSONL 中每行的 text 字段提取成独立 TXT 文件。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 TXT 文件。",
    )
    return parser.parse_args()


def safe_title(title: str, max_length: int = 120) -> str:
    """Return a filename-safe title that also works on Windows."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:max_length].rstrip(" .") or "untitled"

    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"

    return cleaned


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{error}"
                ) from error

            yield line_number, record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    try:
        records = iter_jsonl(args.input)
        for line_number, record in records:
            missing = {key for key in ("page_id", "title", "text") if key not in record}
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"第 {line_number} 行缺少字段：{names}")

            page_id = str(record["page_id"])
            title = safe_title(str(record["title"]))
            text = record["text"]

            if not isinstance(text, str):
                raise ValueError(f"第 {line_number} 行的 text 不是字符串")

            output_file = args.output_dir / f"{page_id}-{title}.txt"
            if output_file.exists() and not args.overwrite:
                skipped += 1
                continue

            output_file.write_text(text, encoding="utf-8")
            written += 1
    except FileNotFoundError:
        raise SystemExit(f"找不到输入文件：{args.input}")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"输入文件：{args.input}")
    print(f"输出目录：{args.output_dir}")
    print(f"写入文件：{written}")
    print(f"跳过文件：{skipped}")


if __name__ == "__main__":
    main()