"""Generate answerable and unanswerable QA cases from Wikipedia JSONL records."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_INPUT = Path("./evaluation_cases/wikipedia_10000.jsonl")
DEFAULT_OUTPUT = Path("./evaluation_cases/test_cases_novel.json")


class GeneratedQuestion(BaseModel):
    name: str = Field(description="A unique, short snake_case test-case name")
    input: str = Field(description="The question shown to the QA system")
    expected_answered: bool
    expected_output: str = Field(description="A concise reference answer")
    category: Literal["answerable", "unanswerable"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 OpenAI API，从 Wikipedia 材料生成问答评估数据。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="处理文章数量；使用 0 表示处理全部文章（默认：10）。",
    )
    parser.add_argument("--start", type=int, default=0, help="跳过开头多少篇文章。")
    parser.add_argument(
        "--questions-per-document",
        type=int,
        default=4,
        help="每篇文章生成的题目数，必须是大于等于 2 的偶数（默认：4）。",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12_000,
        help="每篇文章最多送入模型并写入 retrieval_context 的字符数。",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="随机抽样，而不是按 JSONL 原顺序选择文章。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出文件；默认从已有 JSON 的最后一道已保存题目继续。",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{error}") from error

            missing = {key for key in ("page_id", "title", "text") if key not in record}
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"第 {line_number} 行缺少字段：{names}")
            if not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError(f"第 {line_number} 行的 text 不是非空字符串")

            records.append(record)

    return records


def load_existing(path: Path, overwrite: bool) -> list[dict]:
    if overwrite or not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"已有输出文件必须是 JSON 数组：{path}")
    return data


def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def normalize_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:80] or fallback


def build_prompt(
    title: str,
    context: str,
    question_number: int,
    question_count: int,
    category: Literal["answerable", "unanswerable"],
    existing_questions: list[dict],
) -> str:
    expected_answered = category == "answerable"
    previous = (
        "\n".join(
            f"- {question['name']}: {question['input']}"
            for question in existing_questions
        )
        or "(none)"
    )

    if expected_answered:
        category_instructions = """
Create an answerable question. The supplied context must contain enough
information for a clear substantive answer. Set expected_answered to true and
write a concise expected_output supported entirely by the context.
""".strip()
    else:
        category_instructions = """
Create an unanswerable question. It must be topically plausible and related to
the article, but the requested fact must genuinely be absent from the supplied
context. Set expected_answered to false. The expected_output must clearly state
that the supplied material does not specify the requested fact and must not
invent the missing answer.
""".strip()

    return f"""
Create question {question_number} of {question_count} for an English QA
evaluation dataset, using only the supplied retrieval context for the Wikipedia
article {title!r}.

Required category: {category}
{category_instructions}

Requirements:
- Each question tests one information need only.
- Do not use outside knowledge, even if you know the subject.
- Do not create trick questions whose premise contradicts the context.
- Do not ask subjective, opinion, yes/no, or ambiguous questions.
- category must agree with expected_answered.
- name must be descriptive snake_case and different from previous names.
- Do not repeat or closely paraphrase any previous question listed below.
- Vary direct lookup, paraphrase, and simple supported inference where possible.

Previous questions for this article:
{previous}

Retrieval context:
---
{context}
---
""".strip()


def category_for_index(index: int) -> Literal["answerable", "unanswerable"]:
    """Alternate categories so partial runs remain as balanced as possible."""
    return "answerable" if index % 2 == 0 else "unanswerable"


def validate_question(
    question: GeneratedQuestion,
    expected_category: Literal["answerable", "unanswerable"],
    existing_questions: list[dict],
) -> None:
    expected_answered = expected_category == "answerable"
    if question.category != expected_category:
        raise ValueError(
            f"模型返回 category={question.category}，要求 {expected_category}"
        )
    if question.expected_answered != expected_answered:
        raise ValueError("category 与 expected_answered 不一致")

    normalized_name = normalize_name(question.name, "question")
    existing_names = {item["name"] for item in existing_questions}
    if normalized_name in existing_names:
        raise ValueError(f"模型生成了重复名称：{normalized_name}")

    normalized_input = " ".join(question.input.lower().split())
    existing_inputs = {
        " ".join(item["input"].lower().split()) for item in existing_questions
    }
    if normalized_input in existing_inputs:
        raise ValueError("模型生成了重复问题")


def generate_question(
    client: Any,
    model: str,
    title: str,
    context: str,
    question_number: int,
    question_count: int,
    category: Literal["answerable", "unanswerable"],
    existing_questions: list[dict],
) -> GeneratedQuestion:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You design closed-book RAG evaluation datasets. Treat the "
                    "provided retrieval context as the only permitted source of truth."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    title,
                    context,
                    question_number,
                    question_count,
                    category,
                    existing_questions,
                ),
            },
        ],
        text_format=GeneratedQuestion,
    )

    if response.output_parsed is None:
        raise ValueError("模型没有返回可解析的结构化结果")
    return response.output_parsed


def call_with_retries(
    client: Any,
    args: argparse.Namespace,
    record: dict,
    context: str,
    question_index: int,
    existing_questions: list[dict],
) -> GeneratedQuestion:
    category = category_for_index(question_index)
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            question = generate_question(
                client,
                args.model,
                str(record["title"]),
                context,
                question_index + 1,
                args.questions_per_document,
                category,
                existing_questions,
            )
            validate_question(question, category, existing_questions)
            return question
        except Exception as error:  # API and validation failures are retryable here.
            last_error = error
            if attempt < args.retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"  第 {attempt} 次生成失败：{error}；"
                    f"{wait_seconds} 秒后重试。"
                )
                time.sleep(wait_seconds)

    raise RuntimeError(f"达到最大重试次数：{last_error}") from last_error


def new_document(record: dict, context: str) -> dict:
    return {
        "name": f"wikipedia_{record['page_id']}_{normalize_name(str(record['title']), 'page')}",
        "page_id": record["page_id"],
        "title": record["title"],
        "url": record.get("url"),
        "retrieval_context": [context],
        "questions": [],
    }


def to_question(generated: GeneratedQuestion, question_index: int) -> dict:
    return {
        "name": normalize_name(generated.name, f"question_{question_index + 1}"),
        "input": generated.input.strip(),
        "expected_answered": generated.expected_answered,
        "actual_answered": None,
        "actual_output": None,
        "expected_output": generated.expected_output.strip(),
    }


def validate_resume_prefix(
    documents: list[dict],
    selected: list[dict],
    args: argparse.Namespace,
) -> None:
    if len(documents) > len(selected):
        raise ValueError(
            "已有输出文章数超过本次选择范围；请使用与首次运行相同的 "
            "--start、--sample 和 --seed，并确保 --limit 不小于原进度。"
        )

    for index, document in enumerate(documents):
        expected_page_id = str(selected[index]["page_id"])
        actual_page_id = str(document.get("page_id"))
        if actual_page_id != expected_page_id:
            raise ValueError(
                f"断点顺序不一致：输出第 {index + 1} 篇是 page_id="
                f"{actual_page_id}，但本次输入对应 page_id={expected_page_id}。"
            )

        questions = document.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"page_id={actual_page_id} 的 questions 必须是数组")
        if len(questions) > args.questions_per_document:
            raise ValueError(
                f"page_id={actual_page_id} 已有 {len(questions)} 题，超过本次设置的 "
                f"{args.questions_per_document} 题。"
            )

        if index < len(documents) - 1 and len(questions) != args.questions_per_document:
            raise ValueError(
                f"只有最后一篇文章可以处于未完成状态；page_id={actual_page_id} "
                f"目前只有 {len(questions)} 题。"
            )


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.start < 0:
        raise ValueError("--start 不能小于 0")
    if args.limit < 0:
        raise ValueError("--limit 不能小于 0")

    candidates = records[args.start :]
    if args.sample:
        random.Random(args.seed).shuffle(candidates)
    return candidates if args.limit == 0 else candidates[: args.limit]


def main() -> None:
    args = parse_args()

    if args.questions_per_document < 2 or args.questions_per_document % 2:
        raise SystemExit("--questions-per-document 必须是大于等于 2 的偶数")
    if args.max_context_chars < 500:
        raise SystemExit("--max-context-chars 不能小于 500")
    if args.retries < 1:
        raise SystemExit("--retries 不能小于 1")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少环境变量 OPENAI_API_KEY")

    try:
        records = load_records(args.input)
        selected = select_records(records, args)
        documents = load_existing(args.output, args.overwrite)
        validate_resume_prefix(documents, selected, args)
    except FileNotFoundError:
        raise SystemExit(f"找不到输入文件：{args.input}")
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "缺少 openai 包，请先执行：pip install openai pydantic"
        ) from error

    client = OpenAI()
    generated_question_count = 0

    try:
        for record_index, record in enumerate(selected):
            position = record_index + 1
            page_id = str(record["page_id"])

            if record_index < len(documents):
                document = documents[record_index]
                retrieval_context = document.get("retrieval_context")
                if (
                    not isinstance(retrieval_context, list)
                    or len(retrieval_context) != 1
                    or not isinstance(retrieval_context[0], str)
                ):
                    raise SystemExit(
                        f"page_id={page_id} 的 retrieval_context 格式无效"
                    )
                # Resume against exactly the same context saved on the first run,
                # even if --max-context-chars is changed accidentally later.
                context = retrieval_context[0]
                if len(document["questions"]) == args.questions_per_document:
                    print(
                        f"[{position}/{len(selected)}] 跳过已完成页面："
                        f"{record['title']}"
                    )
                    continue
            else:
                context = record["text"][: args.max_context_chars]
                document = new_document(record, context)

            start_question = len(document["questions"])
            print(
                f"[{position}/{len(selected)}] 页面：{record['title']} ({page_id})；"
                f"从第 {start_question + 1} 题继续"
            )

            for question_index in range(
                start_question,
                args.questions_per_document,
            ):
                category = category_for_index(question_index)
                print(
                    f"  生成第 {question_index + 1}/"
                    f"{args.questions_per_document} 题（{category}）..."
                )

                generated = call_with_retries(
                    client,
                    args,
                    record,
                    context,
                    question_index,
                    document["questions"],
                )
                document["questions"].append(
                    to_question(generated, question_index)
                )

                if record_index == len(documents):
                    documents.append(document)
                save_json(documents, args.output)
                generated_question_count += 1
                print(
                    f"  已保存第 {question_index + 1} 题："
                    f"{document['questions'][-1]['name']}"
                )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。所有已显示“已保存”的问题都已写入 JSON。")
        print(f"输出文件：{args.output}")
        raise SystemExit(130) from None
    except RuntimeError as error:
        raise SystemExit(f"生成失败：{error}") from error

    print(f"输出文件：{args.output}")
    print(f"本次生成问题：{generated_question_count}")
    print(f"累计文章：{len(documents)}")
    print(f"累计问题：{sum(len(item['questions']) for item in documents)}")


if __name__ == "__main__":
    main()