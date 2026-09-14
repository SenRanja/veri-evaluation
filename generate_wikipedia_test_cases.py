"""Generate cited answerable questions, then cross-pair them as unanswerable cases."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.retired-16000.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation_cases" / "test_cases_novel.json"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
ANSWERABLE_QUESTIONS_PER_DOCUMENT = 2
TOTAL_QUESTIONS_PER_DOCUMENT = 4
UNANSWERABLE_REFERENCE = (
    "The supplied retrieval context does not contain the information needed "
    "to answer this question."
)


class GeneratedQuestion(BaseModel):
    name: str = Field(description="A unique, short snake_case test-case name")
    input: str = Field(description="A self-contained answerable question")
    answer: str = Field(description="A concise answer supported by the context")
    evidence_id: int = Field(
        description="The numbered source passage that directly supports the answer"
    )


class GeneratedQuestionBatch(BaseModel):
    questions: list[GeneratedQuestion] = Field(
        description="The requested distinct answerable questions"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从退役题库继承材料和 Veri 文件 ID，让 OpenAI 只生成可回答问题，"
            "再通过跨文章错配构造不可回答问题。"
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="继承案例素材数量；默认 50 个案例，即最终 200 道题。",
    )
    parser.add_argument("--start", type=int, default=0, help="跳过开头多少篇文章。")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有新题库；默认从已保存的可回答题继续。",
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


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"题库必须是 JSON 数组：{path}")
    return data


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def normalize_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:80] or fallback


def normalize_text(value: str) -> str:
    return " ".join(value.split())


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
            if current and len(current) + 1 + len(sentence) > max_chars:
                passages.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            passages.append(current)
    return passages


def validate_source_document(document: dict[str, Any], index: int) -> None:
    missing = {
        key
        for key in ("name", "page_id", "title", "retrieval_context", "veri_file_id")
        if key not in document
    }
    if missing:
        raise ValueError(
            f"源题库第 {index + 1} 篇缺少字段：{', '.join(sorted(missing))}"
        )
    contexts = document["retrieval_context"]
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(not isinstance(context, str) or not context.strip() for context in contexts)
    ):
        raise ValueError(f"源题库第 {index + 1} 篇 retrieval_context 无效")
    if not isinstance(document["veri_file_id"], str) or not document[
        "veri_file_id"
    ].strip():
        raise ValueError(f"源题库第 {index + 1} 篇 veri_file_id 无效")


def select_source_documents(
    source_documents: list[dict[str, Any]], start: int, limit: int
) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start 不能小于 0")
    if limit < 3:
        raise ValueError("--limit 不能小于 3，跨文章错配至少需要 3 篇材料")
    selected = source_documents[start : start + limit]
    if len(selected) != limit:
        raise ValueError(f"源题库只有 {len(selected)} 篇可选，少于 --limit={limit}")
    for index, document in enumerate(selected):
        validate_source_document(document, index)
    return selected


def new_document(source: dict[str, Any]) -> dict[str, Any]:
    document = copy.deepcopy(source)
    document["questions"] = []
    return document


def load_or_initialize_output(
    path: Path,
    selected: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    if overwrite or not path.exists():
        return [new_document(document) for document in selected]

    documents = load_cases(path)
    if len(documents) > len(selected):
        raise ValueError(
            f"已有输出包含 {len(documents)} 篇，当前选择 {len(selected)} 篇；"
            "请确保 --limit 不小于已有进度，或明确使用 --overwrite。"
        )

    for index, (document, source) in enumerate(zip(documents, selected)):
        if str(document.get("page_id")) != str(source["page_id"]):
            raise ValueError(f"已有输出第 {index + 1} 篇与源题库 page_id 不一致")
        if document.get("veri_file_id") != source["veri_file_id"]:
            raise ValueError(f"已有输出第 {index + 1} 篇 veri_file_id 不一致")
        questions = document.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"已有输出第 {index + 1} 篇 questions 不是数组")
        if len(questions) not in (0, 1, 2, 4):
            raise ValueError(f"已有输出第 {index + 1} 篇问题数量无效")

        # Mismatches depend on the full selected set, so rebuild them after expansion.
        if len(questions) == TOTAL_QUESTIONS_PER_DOCUMENT:
            document["questions"] = questions[:ANSWERABLE_QUESTIONS_PER_DOCUMENT]

    documents.extend(
        new_document(source) for source in selected[len(documents) :]
    )
    return documents


def build_prompt(
    title: str,
    passages: list[str],
    question_start_number: int,
    question_count: int,
    existing_questions: list[dict[str, Any]],
    legacy_questions: list[dict[str, Any]],
    validation_feedback: str | None = None,
) -> str:
    previous = (
        "\n".join(f"- {question['input']}" for question in existing_questions)
        or "(none)"
    )
    numbered_context = "\n\n".join(
        f"[E{index}] {passage}" for index, passage in enumerate(passages, 1)
    )
    retired = (
        "\n".join(
            f"- {question.get('name', '')}: {question.get('input', '')}"
            for question in legacy_questions
        )
        or "(none)"
    )
    retry_instruction = (
        f"\nA previous attempt failed validation: {validation_feedback}\n"
        "Generate a new response that explicitly corrects this problem.\n"
        if validation_feedback
        else ""
    )
    return f"""
Create {question_count} distinct answerable question(s) for an English RAG
evaluation dataset about the Wikipedia article {title!r}. These are question
numbers {question_start_number} through
{question_start_number + question_count - 1} of 2. Use only the retrieval context.
{retry_instruction}

Requirements:
- The question must be directly and unambiguously answerable from the context.
- The question must explicitly name the article subject exactly as {title!r} so
  it remains self-contained when paired with another document.
- Ask for one objective fact. Do not ask yes/no, subjective, trick, or ambiguous
  questions, and do not require outside knowledge.
- Give a concise substantive answer.
- evidence_id must be the integer from the [E<number>] passage that most directly
    supports the answer. Do not include the E prefix.
- Use a descriptive snake_case name that differs from previous names.
- Do not repeat or closely paraphrase a previous question.
- This replaces a retired dataset. Do not reuse any retired name or question.

Previous questions for this article:
{previous}

Retired questions that must not be reused:
{retired}

Retrieval context:
---
{numbered_context}
---
""".strip()


def validate_generated_question(
    question: GeneratedQuestion,
    title: str,
    passages: list[str],
    existing_questions: list[dict[str, Any]],
    legacy_questions: list[dict[str, Any]],
) -> None:
    if not question.input.strip() or not question.answer.strip():
        raise ValueError("模型返回了空问题或空答案")
    if normalize_text(title).casefold() not in normalize_text(question.input).casefold():
        raise ValueError("问题没有明确写出文章标题")

    normalized_name = normalize_name(question.name, "question")
    if normalized_name in {item["name"] for item in existing_questions}:
        raise ValueError(f"模型生成了重复名称：{normalized_name}")
    legacy_names = {
        normalize_name(str(item.get("name", "")), "legacy_question")
        for item in legacy_questions
    }
    if normalized_name in legacy_names:
        raise ValueError(f"模型复用了退役题目名称：{normalized_name}")
    normalized_input = normalize_text(question.input).casefold()
    if normalized_input in {
        normalize_text(item["input"]).casefold() for item in existing_questions
    }:
        raise ValueError("模型生成了重复问题")
    legacy_inputs = {
        normalize_text(str(item.get("input", ""))).casefold()
        for item in legacy_questions
    }
    if normalized_input in legacy_inputs:
        raise ValueError("模型复用了退役问题")

    if not 1 <= question.evidence_id <= len(passages):
        raise ValueError(
            f"evidence_id={question.evidence_id} 超出 1..{len(passages)} 范围"
        )


def generate_questions(
    client: Any,
    model: str,
    title: str,
    passages: list[str],
    question_start_number: int,
    question_count: int,
    existing_questions: list[dict[str, Any]],
    legacy_questions: list[dict[str, Any]],
    validation_feedback: str | None = None,
) -> GeneratedQuestionBatch:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You create reliable RAG test questions. Every generated "
                    "question must be answerable and linked to the numbered source "
                    "passage that directly supports it."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    title,
                    passages,
                    question_start_number,
                    question_count,
                    existing_questions,
                    legacy_questions,
                    validation_feedback,
                ),
            },
        ],
        text_format=GeneratedQuestionBatch,
    )
    if response.output_parsed is None:
        raise ValueError("模型没有返回可解析的结构化结果")
    return response.output_parsed


def call_with_retries(
    client: Any,
    args: argparse.Namespace,
    document: dict[str, Any],
    legacy_questions: list[dict[str, Any]],
) -> list[GeneratedQuestion]:
    context = "\n\n".join(document["retrieval_context"])
    passages = split_evidence_passages(context)
    if not passages:
        raise RuntimeError("retrieval_context 无法切分出证据段")
    question_start_number = len(document["questions"]) + 1
    question_count = ANSWERABLE_QUESTIONS_PER_DOCUMENT - len(document["questions"])
    last_error: Exception | None = None
    validation_feedback = None
    for attempt in range(1, args.retries + 1):
        try:
            batch = generate_questions(
                client,
                args.model,
                str(document["title"]),
                passages,
                question_start_number,
                question_count,
                document["questions"],
                legacy_questions,
                validation_feedback,
            )
            if len(batch.questions) != question_count:
                raise ValueError(
                    f"模型返回 {len(batch.questions)} 题，要求 {question_count} 题"
                )
            validated_questions = list(document["questions"])
            for question in batch.questions:
                validate_generated_question(
                    question,
                    str(document["title"]),
                    passages,
                    validated_questions,
                    legacy_questions,
                )
                validated_questions.append(
                    {
                        "name": normalize_name(question.name, "question"),
                        "input": question.input.strip(),
                    }
                )
            return batch.questions
        except Exception as error:  # API and validation failures are retryable.
            last_error = error
            validation_feedback = str(error)
            if attempt < args.retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"  第 {attempt} 次生成失败：{error}；"
                    f"{wait_seconds} 秒后重试。"
                )
                time.sleep(wait_seconds)
    raise RuntimeError(f"达到最大重试次数：{last_error}") from last_error


def to_answerable_question(
    generated: GeneratedQuestion, question_index: int, citation: str
) -> dict[str, Any]:
    answer = generated.answer.strip()
    return {
        "name": normalize_name(generated.name, f"answerable_{question_index + 1}"),
        "input": generated.input.strip(),
        "expected_answered": True,
        "actual_answered": None,
        "actual_output": None,
        "expected_output": f'{answer}\n\nSource citation: "{citation}"',
        "reference_citation": citation,
    }


def find_mismatch_sources(
    documents: list[dict[str, Any]], target_index: int
) -> list[int]:
    target_context = normalize_text(
        "\n\n".join(documents[target_index]["retrieval_context"])
    ).casefold()
    sources = []
    for offset in range(1, len(documents)):
        source_index = (target_index + offset) % len(documents)
        source_title = normalize_text(str(documents[source_index]["title"])).casefold()
        if source_title and source_title not in target_context:
            sources.append(source_index)
        if len(sources) == ANSWERABLE_QUESTIONS_PER_DOCUMENT:
            return sources
    raise ValueError(
        f"无法为 page_id={documents[target_index]['page_id']} 找到两个安全错配来源"
    )


def add_mismatched_questions(documents: list[dict[str, Any]]) -> None:
    if any(
        len(document["questions"]) != ANSWERABLE_QUESTIONS_PER_DOCUMENT
        for document in documents
    ):
        raise ValueError("必须先为每篇文章生成两道可回答问题，才能执行错配")

    answerable_questions = [
        copy.deepcopy(document["questions"]) for document in documents
    ]
    for target_index, document in enumerate(documents):
        for source_question_index, source_index in enumerate(
            find_mismatch_sources(documents, target_index)
        ):
            source_document = documents[source_index]
            source_question = answerable_questions[source_index][source_question_index]
            document["questions"].append(
                {
                    "name": normalize_name(
                        f"mismatched_{source_document['page_id']}_"
                        f"{source_question['name']}",
                        f"mismatched_{source_index}_{source_question_index}",
                    ),
                    "input": source_question["input"],
                    "expected_answered": False,
                    "actual_answered": None,
                    "actual_output": None,
                    "expected_output": UNANSWERABLE_REFERENCE,
                    "mismatched_from_page_id": source_document["page_id"],
                    "mismatched_from_title": source_document["title"],
                }
            )


def remove_legacy_overlaps(
    documents: list[dict[str, Any]], source_documents: list[dict[str, Any]]
) -> int:
    overlapping: set[tuple[int, int]] = set()
    for document_index, (document, source) in enumerate(
        zip(documents, source_documents, strict=True)
    ):
        legacy_questions = source.get("questions", [])
        legacy_names = {
            normalize_name(str(item.get("name", "")), "legacy_question")
            for item in legacy_questions
        }
        legacy_inputs = {
            normalize_text(str(item.get("input", ""))).casefold()
            for item in legacy_questions
        }
        for question_index, question in enumerate(
            document.get("questions", [])[:ANSWERABLE_QUESTIONS_PER_DOCUMENT]
        ):
            name = normalize_name(str(question.get("name", "")), "question")
            question_input = normalize_text(str(question.get("input", ""))).casefold()
            if name in legacy_names or question_input in legacy_inputs:
                overlapping.add((document_index, question_index))

    if not overlapping:
        return 0

    for document_index, document in enumerate(documents):
        answerable = document["questions"][:ANSWERABLE_QUESTIONS_PER_DOCUMENT]
        document["questions"] = [
            question
            for question_index, question in enumerate(answerable)
            if (document_index, question_index) not in overlapping
        ]
    return len(overlapping)


def main() -> None:
    args = parse_args()
    if args.retries < 1:
        raise SystemExit("--retries 不能小于 1")

    try:
        load_env_file(args.env_file)
        source_documents = load_cases(args.source)
        selected = select_source_documents(source_documents, args.start, args.limit)
        documents = load_or_initialize_output(args.output, selected, args.overwrite)
    except FileNotFoundError as error:
        raise SystemExit(f"找不到文件：{error.filename}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    removed_overlap_count = remove_legacy_overlaps(documents, selected)
    if removed_overlap_count:
        save_json(documents, args.output)
        print(f"已移除与退役题库重合的可回答题：{removed_overlap_count}")

    if all(
        len(document["questions"]) == TOTAL_QUESTIONS_PER_DOCUMENT
        for document in documents
    ):
        print(f"题库已完成：{args.output}")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(f"OPENAI_API_KEY 未设置或不存在于 {args.env_file}")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("缺少 openai 包，请先执行：pip install openai pydantic") from error

    save_json(documents, args.output)
    client = OpenAI()
    generated_question_count = 0
    failed_documents = []
    try:
        for document_index, document in enumerate(documents):
            position = document_index + 1
            if len(document["questions"]) < ANSWERABLE_QUESTIONS_PER_DOCUMENT:
                question_start = len(document["questions"]) + 1
                question_count = (
                    ANSWERABLE_QUESTIONS_PER_DOCUMENT - len(document["questions"])
                )
                print(
                    f"[{position}/{len(documents)}] {document['title']}："
                    f"一次生成 {question_count} 道可回答题"
                    f"（第 {question_start}-2 题）..."
                )
                try:
                    generated_batch = call_with_retries(
                        client,
                        args,
                        document,
                        selected[document_index].get("questions", []),
                    )
                    context = "\n\n".join(document["retrieval_context"])
                    passages = split_evidence_passages(context)
                    new_questions = [
                        to_answerable_question(
                            generated,
                            question_start - 1 + batch_index,
                            passages[generated.evidence_id - 1],
                        )
                        for batch_index, generated in enumerate(generated_batch)
                    ]
                    document["questions"].extend(new_questions)
                    save_json(documents, args.output)
                    generated_question_count += len(new_questions)
                    print(
                        "  已保存："
                        + ", ".join(question["name"] for question in new_questions)
                    )
                except RuntimeError as error:
                    failed_documents.append((position, str(error)))
                    print(f"  此案例生成失败，保留缺口并继续：{error}")
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。所有已显示“已保存”的问题都已写入 JSON。")
        print(f"输出文件：{args.output}")
        raise SystemExit(130) from None
    incomplete = [
        index + 1
        for index, document in enumerate(documents)
        if len(document["questions"]) != ANSWERABLE_QUESTIONS_PER_DOCUMENT
    ]
    if not incomplete:
        add_mismatched_questions(documents)
        save_json(documents, args.output)
    print(f"输出文件：{args.output}")
    print(f"本次调用模型生成可回答题：{generated_question_count}")
    print(f"累计文章：{len(documents)}")
    print(f"累计问题：{sum(len(item['questions']) for item in documents)}")
    if incomplete:
        print(f"尚有 {len(incomplete)} 个案例未完成；重新运行同一命令即可补齐。")
        print("未完成案例序号：" + ", ".join(map(str, incomplete)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()