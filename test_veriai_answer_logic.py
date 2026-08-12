import pytest

from veriai_answer import (
    add_source_index,
    build_question_prompt,
    extract_answer_section,
    extract_response_text,
    is_answered,
    normalize_citation_output,
)


@pytest.mark.parametrize(
    "answer",
    [
        "【未搜尋到知識庫中的相關資訊】",
        "目前選出的段落中並沒有提到有哪些航空公司營運。",
        "The supplied material does not specify the requested information.",
        "The provided text does not list the requested patent titles.",
        "真意查询失败",
        "【Cited passage】\nNo relevant passage",
    ],
)
def test_is_answered_recognizes_refusals(answer: str) -> None:
    assert is_answered(answer) is False


def test_is_answered_accepts_substantive_answer() -> None:
    assert is_answered("The airport opened on May 19, 2022.") is True


def test_extract_response_text_reads_openai_chat_shape() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "  Wireless communications.  ",
                }
            }
        ]
    }

    assert extract_response_text(payload) == "Wireless communications."


def test_extract_response_text_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="did not contain answer text"):
        extract_response_text({"choices": []})


def test_build_question_prompt_requires_exact_passage_and_source() -> None:
    prompt = build_question_prompt(
        "What field does Ahmad Bazzi specialize in?",
        "Ahmad Bazzi",
        "70533387-Ahmad Bazzi.txt",
    )

    assert "【Answer】" in prompt
    assert "【Cited passage】" in prompt
    assert "【Source】" in prompt
    assert "quote the supporting passage exactly" in prompt.casefold()
    assert "70533387-Ahmad Bazzi.txt" in prompt


def test_normalize_citation_output_expands_short_refusal() -> None:
    answer = normalize_citation_output(
        "【未搜尋到知識庫中的相關資訊】",
        "Ahmad Bazzi",
        "70533387-Ahmad Bazzi.txt",
    )

    assert extract_answer_section(answer) == "【未搜尋到知識庫中的相關資訊】"
    assert "【Cited passage】\nNo relevant passage" in answer
    assert "70533387-Ahmad Bazzi.txt" in answer
    assert is_answered(answer) is False


def test_is_answered_ignores_no_relevant_passage_outside_answer_section() -> None:
    answer = (
        "【Answer】\nYouTube\n"
        "【Cited passage】\nNo relevant passage is placeholder text.\n"
        "【Source】\nAhmad Bazzi (filename: 70533387-Ahmad Bazzi.txt)"
    )

    assert is_answered(answer) is True


def test_add_source_index_appends_file_metadata_once() -> None:
    answer = add_source_index(
        "【Answer】\nYouTube",
        "file-123",
        "Ahmad Bazzi",
        "70533387-Ahmad Bazzi.txt",
    )

    assert "【Source index】" in answer
    assert "file_id: file-123" in answer
    assert "filename: 70533387-Ahmad Bazzi.txt" in answer
    assert add_source_index(
        answer,
        "file-123",
        "Ahmad Bazzi",
        "70533387-Ahmad Bazzi.txt",
    ) == answer