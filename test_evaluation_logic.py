import asyncio
import json

from evaluation import (
    build_decision_summary,
    build_quality_summary,
    evaluate_case,
    get_decision_state,
    load_cases,
)


class FakeMetric:
    def __init__(self, name, score, threshold=0.7):
        self.name = name
        self.score = score
        self.threshold = threshold
        self.reason = None

    async def a_measure(self, test_case, _show_indicator=False):
        return self.score


def make_case(expected_answered, actual_answered):
    return {
        "document": "document",
        "name": "question",
        "input": "Question?",
        "expected_output": "Expected.",
        "actual_output": "Actual.",
        "retrieval_context": ["Context."],
        "expected_answered": expected_answered,
        "actual_answered": actual_answered,
    }


def make_metrics(correctness, faithfulness, answer_relevancy, contextual=0.0):
    return [
        FakeMetric("Contextual Relevancy", contextual),
        FakeMetric("Answer Relevancy", answer_relevancy),
        FakeMetric("Correctness", correctness),
        FakeMetric("Faithfulness", faithfulness),
    ]


def test_decision_states():
    assert get_decision_state(make_case(True, True)) == "AA"
    assert get_decision_state(make_case(False, False)) == "NN"
    assert get_decision_state(make_case(True, False)) == "AN"
    assert get_decision_state(make_case(False, True)) == "NA"


def test_metric_applicability_for_answer_and_refusal():
    nn_result = asyncio.run(
        evaluate_case(make_case(False, False), make_metrics(1.0, 0.0, 0.0))
    )
    aa_result = asyncio.run(
        evaluate_case(make_case(True, True), make_metrics(1.0, 0.0, 1.0))
    )

    assert nn_result["case_passed"] is True
    assert aa_result["case_passed"] is False
    assert nn_result["passed"] == nn_result["case_passed"]
    assert next(
        metric for metric in nn_result["metrics"]
        if metric["name"] == "Contextual Relevancy"
    )["gates_case"] is False


def test_model_specific_fields_take_priority_with_legacy_fallback(tmp_path):
    question = {
        "name": "question",
        "input": "Question?",
        "expected_answered": True,
        "expected_output": "Expected.",
        "actual_answered": False,
        "actual_output": "Legacy.",
        "actual_answered_target-model": True,
        "actual_output_target-model": "Preferred.",
    }
    document = {
        "name": "document",
        "retrieval_context": ["Context."],
        "questions": [question],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([document]), encoding="utf-8")

    assert load_cases(path, "target-model")[0]["actual_output"] == "Preferred."

    del question["actual_answered_target-model"]
    del question["actual_output_target-model"]
    path.write_text(json.dumps([document]), encoding="utf-8")
    assert load_cases(path, "target-model")[0]["actual_output"] == "Legacy."


def test_conditional_summaries_and_zero_denominators():
    aa_result = asyncio.run(
        evaluate_case(make_case(True, True), make_metrics(0.9, 0.8, 0.7))
    )
    an_result = asyncio.run(
        evaluate_case(make_case(True, False), make_metrics(1.0, 0.0, 0.0))
    )
    nn_result = asyncio.run(
        evaluate_case(make_case(False, False), make_metrics(0.8, 0.0, 0.0))
    )
    results = [aa_result, an_result, nn_result]

    decision = build_decision_summary(results)
    quality = build_quality_summary(results)

    assert decision["rate_details"]["decision_accuracy"] == {
        "numerator": 2,
        "denominator": 3,
        "value": 2 / 3,
    }
    assert quality["aa_mean_faithfulness"]["value"] == 0.8
    assert quality["nn_mean_correctness"]["value"] == 0.8
    assert quality["correct_answer_rate"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert build_quality_summary([])["correct_answer_rate"]["value"] is None