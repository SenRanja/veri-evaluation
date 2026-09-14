from copy import deepcopy

import pytest

from generate_wikipedia_test_cases import (
    GeneratedQuestion,
    add_mismatched_questions,
    build_prompt,
    load_or_initialize_output,
    new_document,
    remove_legacy_overlaps,
    split_evidence_passages,
    to_answerable_question,
    validate_generated_question,
)


def source_document(page_id: int, title: str, fact: str) -> dict:
    return {
        "name": f"wikipedia_{page_id}",
        "page_id": page_id,
        "title": title,
        "url": f"https://example.test/{page_id}",
        "retrieval_context": [f"{title} is the subject. {fact}"],
        "veri_file_id": f"file-{page_id}",
        "questions": [{"old": "question"}],
    }


def answerable_question(title: str, fact: str, index: int) -> dict:
    generated = GeneratedQuestion(
        name=f"fact_{index}",
        input=f"What fact is stated about {title}?",
        answer=fact,
        evidence_id=1,
    )
    return to_answerable_question(generated, index, fact)


def test_new_document_inherits_material_and_file_id_but_resets_questions():
    source = source_document(1, "Alpha", "Alpha has one fact.")

    result = new_document(source)

    assert result["veri_file_id"] == "file-1"
    assert result["retrieval_context"] == source["retrieval_context"]
    assert result["questions"] == []
    assert source["questions"] == [{"old": "question"}]


def test_generated_question_requires_title_and_valid_evidence_id():
    context = "Alpha is the subject. Alpha has one exact fact."
    passages = split_evidence_passages(context)
    valid = GeneratedQuestion(
        name="alpha_fact",
        input="What exact fact is stated about Alpha?",
        answer="It has one exact fact.",
        evidence_id=1,
    )
    validate_generated_question(valid, "Alpha", passages, [], [])

    invalid = valid.model_copy(update={"evidence_id": 2})
    with pytest.raises(ValueError, match="超出"):
        validate_generated_question(invalid, "Alpha", passages, [], [])

    assert passages[valid.evidence_id - 1] in context


def test_retry_prompt_includes_validation_feedback():
    prompt = build_prompt(
        "Alpha",
        ["Alpha has one exact fact."],
        1,
        1,
        [],
        [],
        "evidence_id 超出范围",
    )

    assert "A previous attempt failed validation" in prompt
    assert "evidence_id 超出范围" in prompt


def test_existing_output_expands_and_removes_old_mismatches(tmp_path):
    sources = [
        source_document(1, "Alpha", "Alpha has one fact."),
        source_document(2, "Beta", "Beta has a second fact."),
        source_document(3, "Gamma", "Gamma has a third fact."),
    ]
    completed = [new_document(source) for source in sources]
    for document, fact in zip(
        completed,
        (
            "Alpha has one fact.",
            "Beta has a second fact.",
            "Gamma has a third fact.",
        ),
        strict=True,
    ):
        document["questions"] = [
            answerable_question(document["title"], fact, 0),
            answerable_question(document["title"], fact, 1),
        ]
    # Simulate a completed smaller run; expansion must discard stale mismatches.
    add_mismatched_questions(completed)
    existing = completed[:2]
    path = tmp_path / "cases.json"
    path.write_text(__import__("json").dumps(existing), encoding="utf-8")

    expanded = load_or_initialize_output(path, sources, overwrite=False)

    assert len(expanded) == 3
    assert [len(document["questions"]) for document in expanded] == [2, 2, 0]


def test_remove_legacy_overlaps_drops_stale_mismatches():
    sources = [
        source_document(1, "Alpha", "Alpha has one fact."),
        source_document(2, "Beta", "Beta has a second fact."),
        source_document(3, "Gamma", "Gamma has a third fact."),
    ]
    documents = [new_document(source) for source in sources]
    for document, fact in zip(
        documents,
        ("Alpha has one fact.", "Beta has a second fact.", "Gamma has a third fact."),
        strict=True,
    ):
        document["questions"] = [
            answerable_question(document["title"], fact, 0),
            answerable_question(document["title"], fact, 1),
        ]
    add_mismatched_questions(documents)
    sources[0]["questions"] = [
        {
            "name": documents[0]["questions"][0]["name"],
            "input": "A different retired question?",
        }
    ]

    removed = remove_legacy_overlaps(documents, sources)

    assert removed == 1
    assert len(documents[0]["questions"]) == 1
    assert all(len(document["questions"]) == 2 for document in documents[1:])


def test_cross_pairing_creates_two_answerable_and_two_unanswerable_questions():
    documents = [
        new_document(source_document(1, "Alpha", "Alpha has one fact.")),
        new_document(source_document(2, "Beta", "Beta has a second fact.")),
        new_document(source_document(3, "Gamma", "Gamma has a third fact.")),
    ]
    facts = ["Alpha has one fact.", "Beta has a second fact.", "Gamma has a third fact."]
    for document, fact in zip(documents, facts, strict=True):
        document["questions"] = [
            answerable_question(document["title"], fact, 0),
            answerable_question(document["title"], fact, 1),
        ]

    before = deepcopy(documents)
    add_mismatched_questions(documents)

    for document, original in zip(documents, before, strict=True):
        assert document["questions"][:2] == original["questions"]
        assert [question["expected_answered"] for question in document["questions"]] == [
            True,
            True,
            False,
            False,
        ]
        mismatch_sources = {
            question["mismatched_from_page_id"] for question in document["questions"][2:]
        }
        assert len(mismatch_sources) == 2
        assert document["page_id"] not in mismatch_sources