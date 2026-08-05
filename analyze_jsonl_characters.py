"""Calculate per-line character-count statistics for a JSONL file."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path


DEFAULT_PATH = Path("./evaluation_cases/wikipedia_10000.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 JSONL 每行的字符数（不包含行尾换行符）。"
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"JSONL 文件路径（默认：{DEFAULT_PATH}）",
    )
    return parser.parse_args()


def character_count(line: str) -> int:
    """Count Unicode characters, excluding only the line-ending characters."""
    return len(line.rstrip("\r\n"))


def main() -> None:
    args = parse_args()

    try:
        with args.path.open("r", encoding="utf-8") as file:
            counts = [character_count(line) for line in file]
    except FileNotFoundError:
        raise SystemExit(f"找不到文件：{args.path}")
    except UnicodeDecodeError as error:
        raise SystemExit(f"文件不是有效的 UTF-8 文本：{error}") from error

    if not counts:
        raise SystemExit(f"文件为空：{args.path}")

    frequencies = Counter(counts)
    highest_frequency = max(frequencies.values())
    modes = sorted(
        count for count, frequency in frequencies.items()
        if frequency == highest_frequency
    )

    longest = max(counts)
    shortest = min(counts)
    longest_lines = [index for index, count in enumerate(counts, 1) if count == longest]
    shortest_lines = [index for index, count in enumerate(counts, 1) if count == shortest]

    print(f"文件：{args.path}")
    print(f"总行数：{len(counts)}")
    print(f"平均数：{statistics.fmean(counts):.2f}")
    print(f"最长：{longest} characters（行号：{', '.join(map(str, longest_lines))}）")
    print(f"最短：{shortest} characters（行号：{', '.join(map(str, shortest_lines))}）")
    print(f"中位数：{statistics.median(counts):g}")
    print(f"众数：{', '.join(map(str, modes))}（各出现 {highest_frequency} 次）")


if __name__ == "__main__":
    main()