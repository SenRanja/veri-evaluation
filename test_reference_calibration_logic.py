import json
from types import SimpleNamespace

from calibrate_reference_answers import (
    ReferenceCalibration,
    apply_completed_reviews,
    build_disagreement_candidates,
    calibration_settings,
    review_candidate,
)


def question(gpt_answered: bool, deepseek_answered: bool) -> dict:
    return {
        "name": "question",
        "input": "When was Alpha founded?",
        "expected_answered": False,
        "expected_output": "Not specified.",
        "actual_answered_gpt-4o-mini": gpt_answered,
        "actual_output_gpt-4o-mini": "GPT answer.",
        "actual_answered_deepseek": deepseek_answered,
        "actual_output_deepseek": "DeepSeek answer.",
    }


def document(questions: list[dict]) -> dict:
    return {
        "name": "document",
        "title": "Alpha",
        "retrieval_context": ["Alpha was founded in 1999."],
        "questions": questions,
    }


def test_calibration_settings_supports_deepseek_workers() -> None:
    settings = calibration_settings(
        {
            "reference_calibration": {
                "model": "deepseek-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
                "prompt": "Review the answer.",
                "max_workers": 4,
            }
        }
    )

    assert settings["max_workers"] == 4


def test_only_gpt_deepseek_answerability_disagreements_are_selected() -> None:
    disagree = question(True, False)
    agree = {**question(True, True), "name": "agree"}
    incomplete = {**question(True, False), "name": "incomplete"}
    del incomplete["actual_output_deepseek"]

    candidates = build_disagreement_candidates(
        [document([disagree, agree, incomplete])]
    )

    assert [candidate["question"]["name"] for candidate in candidates] == [
        "question"
    ]


def test_deepseek_review_uses_both_answers_and_builds_exact_citation() -> None:
    revision = ReferenceCalibration(
        expected_answered=True,
        answer="Alpha was founded in 1999.",
        evidence_id=1,
        ambiguous_question=False,
        confidence="high",
        rationale="The context directly states the date.",
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(revision.model_dump())
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    candidate = build_disagreement_candidates(
        [document([question(True, False)])]
    )[0]

    result, expected_output = review_candidate(
        client, "deepseek-flash", "Calibrate the golden answer.", candidate
    )

    assert result is not None
    assert 'Source citation: "Alpha was founded in 1999."' in expected_output
    prompt = completions.request["messages"][0]["content"]
    assert "GPT answer." in prompt
    assert "DeepSeek answer." in prompt
    assert "Authoritative retrieval context" in prompt


def test_apply_updates_only_high_confidence_non_ambiguous_reviews() -> None:
    questions = [question(True, False)]
    documents = [document(questions)]
    audit = {
        "reviews": [
            {
                "status": "completed",
                "document": "document",
                "name": "question",
                "previous": {
                    "expected_answered": False,
                    "expected_output": "Not specified.",
                },
                "revision": {
                    "expected_answered": True,
                    "corrected_expected_output": (
                        'Alpha was founded in 1999.\n\nSource citation: '
                        '"Alpha was founded in 1999."'
                    ),
                    "confidence": "high",
                    "ambiguous_question": False,
                },
                "applied": False,
            }
        ]
    }

    counts = apply_completed_reviews(documents, audit)

    assert counts == {"updated": 1}
    assert questions[0]["expected_answered"] is True
    assert "Source citation" in questions[0]["expected_output"]
    assert questions[0]["actual_output_gpt-4o-mini"] == "GPT answer."