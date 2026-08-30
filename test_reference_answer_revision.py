from types import SimpleNamespace

from revise_reference_answers import (
    ReferenceRevision,
    apply_completed_reviews,
    candidate_reasons,
    load_or_create_audit,
    review_candidate,
)


def model_case(state: str, answered: bool, expected: bool) -> dict:
    return {
        "decision_state": state,
        "actual_answered": answered,
        "expected_answered": expected,
        "actual_output": f"answer for {state}",
    }


def all_models(case: dict) -> dict[str, dict]:
    return {
        "gpt-4o-mini": dict(case),
        "gemini-3.5-flash": dict(case),
        "veri": dict(case),
    }


def test_candidate_reasons_include_disagreement_and_unanimous_mismatches() -> None:
    disagreement = {
        "gpt-4o-mini": model_case("NN", False, False),
        "gemini-3.5-flash": model_case("NA", True, False),
        "veri": model_case("NA", True, False),
    }
    unanimous_na = all_models(model_case("NA", True, False))
    unanimous_an = all_models(model_case("AN", False, True))
    unanimous_aa = all_models(model_case("AA", True, True))
    without_gemini = {
        "gpt-4o-mini": model_case("NN", False, False),
        "veri": model_case("NA", True, False),
    }
    only_veri = {"veri": model_case("NA", True, False)}

    assert candidate_reasons(disagreement, False) == [
        "model_decision_disagreement"
    ]
    assert candidate_reasons(unanimous_na, False) == ["unanimous_NA"]
    assert candidate_reasons(unanimous_an, False) == ["unanimous_AN"]
    assert candidate_reasons(unanimous_na, True) == []
    assert candidate_reasons(unanimous_aa, False) == []
    assert candidate_reasons(without_gemini, False) == [
        "model_decision_disagreement"
    ]
    assert candidate_reasons(only_veri, False) == []


class FakeResponses:
    def __init__(self, revision: ReferenceRevision):
        self.revision = revision
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_parsed=self.revision)


def test_review_candidate_sends_file_context_and_all_model_answers() -> None:
    revision = ReferenceRevision(
        retrieval_context_answerable=True,
        corrected_expected_output="The supported answer.",
        retrieval_context_evidence=["supported evidence"],
        full_document_answerable=True,
        full_document_answer="The supported answer.",
        scope_mismatch=False,
        ambiguous_question=False,
        confidence="high",
        needs_human_review=False,
        rationale="The evidence directly answers the question.",
    )
    responses = FakeResponses(revision)
    client = SimpleNamespace(responses=responses)
    candidate = {
        "document": {
            "title": "Example",
            "retrieval_context": ["supported evidence"],
        },
        "question": {
            "input": "What is supported?",
            "expected_answered": False,
            "expected_output": "Not specified.",
        },
        "model_cases": {
            model: model_case("NA", True, False)
            for model in ("gpt-4o-mini", "veri")
        },
    }

    result = review_candidate(client, "gpt-4o-mini", "file-123", candidate)

    assert result is revision
    request = responses.request
    assert request["model"] == "gpt-4o-mini"
    content = request["input"][1]["content"]
    assert content[0] == {"type": "input_file", "file_id": "file-123"}
    prompt = content[1]["text"]
    assert "supported evidence" in prompt
    assert "gpt-4o-mini" in prompt
    assert "gemini-3.5-flash" in prompt
    assert "result: unavailable" in prompt
    assert "veri" in prompt
    assert "authoritative scope" in prompt


def test_load_or_create_audit_migrates_all_model_selection(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    cases_path = tmp_path / "cases.json"
    result_files = {
        model: tmp_path / f"{model}.json"
        for model in ("gpt-4o-mini", "gemini-3.5-flash", "veri")
    }
    audit_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": 1,
                "cases_file": str(cases_path),
                "judge_model": "gpt-4o",
                "selection": {
                    "requires_all_models": [
                        "gpt-4o-mini",
                        "gemini-3.5-flash",
                        "veri",
                    ],
                    "disagreements_only": False,
                },
                "result_files": {
                    model: str(path) for model, path in result_files.items()
                },
                "uploaded_files": {},
                "reviews": [{"status": "completed"}],
            }
        ),
        encoding="utf-8",
    )

    audit = load_or_create_audit(
        audit_path,
        cases_path,
        result_files,
        "gpt-4o",
        False,
    )

    assert audit["schema_version"] == 2
    assert audit["selection"]["minimum_model_results"] == 2
    assert audit["reviews"] == [{"status": "completed"}]


def test_apply_completed_reviews_updates_only_eligible_golden_fields() -> None:
    documents = [
        {
            "name": "doc",
            "questions": [
                {
                    "name": "eligible",
                    "expected_answered": False,
                    "expected_output": "Not specified.",
                    "actual_output_veri": "Preserve this model output.",
                },
                {
                    "name": "uncertain",
                    "expected_answered": False,
                    "expected_output": "Still not specified.",
                },
            ],
        }
    ]
    audit = {
        "reviews": [
            {
                "status": "completed",
                "document": "doc",
                "name": "eligible",
                "previous": {
                    "expected_answered": False,
                    "expected_output": "Not specified.",
                },
                "revision": {
                    "retrieval_context_answerable": True,
                    "corrected_expected_output": "Correct answer.",
                    "confidence": "high",
                    "needs_human_review": False,
                    "ambiguous_question": False,
                },
                "applied": False,
            },
            {
                "status": "completed",
                "document": "doc",
                "name": "uncertain",
                "previous": {
                    "expected_answered": False,
                    "expected_output": "Still not specified.",
                },
                "revision": {
                    "retrieval_context_answerable": True,
                    "corrected_expected_output": "Possible answer.",
                    "confidence": "medium",
                    "needs_human_review": True,
                    "ambiguous_question": False,
                },
                "applied": False,
            },
        ]
    }

    counts = apply_completed_reviews(documents, audit)

    eligible, uncertain = documents[0]["questions"]
    assert eligible["expected_answered"] is True
    assert eligible["expected_output"] == "Correct answer."
    assert eligible["actual_output_veri"] == "Preserve this model output."
    assert uncertain["expected_answered"] is False
    assert counts == {"updated": 1, "held_for_human_review": 1}