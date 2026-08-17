import asyncio
import json

import evaluation
from evaluation import (
    build_decision_summary,
    build_live_report,
    build_metrics,
    build_quality_summary,
    evaluate_case,
    evaluate_cases,
    get_metric_identity,
    get_decision_state,
    get_target_models,
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


def test_every_metric_response_produces_a_snapshot():
    snapshots = []

    async def capture(result):
        snapshots.append(json.loads(json.dumps(result)))

    result = asyncio.run(
        evaluate_case(
            make_case(True, True),
            make_metrics(1.0, 1.0, 1.0, contextual=0.0),
            on_metric_response=capture,
        )
    )

    assert [len(snapshot["metrics"]) for snapshot in snapshots] == [1, 2, 3, 4]
    assert [snapshot["status"] for snapshot in snapshots] == [
        "in_progress",
        "in_progress",
        "in_progress",
        "completed",
    ]
    assert result["case_passed"] is True


def test_real_metrics_use_stable_ids():
    config = {
        "judge": {"model": "gpt-4o-mini"},
        "metrics": {
            "correctness": {"threshold": 0.7, "criteria": "Be correct."},
            "faithfulness": {"threshold": 0.7},
            "answer_relevancy": {"threshold": 0.7},
            "contextual_relevancy": {"threshold": 0.7},
        },
    }

    assert [
        get_metric_identity(metric) for metric in build_metrics(config)
    ] == [
        ("contextual_relevancy", "Contextual Relevancy"),
        ("answer_relevancy", "Answer Relevancy"),
        ("correctness", "Correctness"),
        ("faithfulness", "Faithfulness"),
    ]


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


def test_target_models_supports_dual_targets_and_legacy_single_target():
    assert get_target_models(
        {"target": {"models": ["gpt-4o-mini", "veri"]}}
    ) == ["gpt-4o-mini", "veri"]
    assert get_target_models({"target": {"model": "gpt-4o-mini"}}) == [
        "gpt-4o-mini"
    ]


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


def test_live_report_excludes_partial_cases_from_summaries():
    completed = asyncio.run(
        evaluate_case(make_case(True, True), make_metrics(0.9, 0.8, 0.7))
    )
    partial = {
        **completed,
        "name": "partial",
        "status": "in_progress",
        "case_passed": None,
        "passed": None,
        "metrics": completed["metrics"][:1],
    }

    report = build_live_report([completed, partial], 2, 4)

    assert report["progress"] == {
        "status": "running",
        "total_cases": 2,
        "completed_cases": 1,
        "metric_responses": 5,
        "expected_metric_responses": 8,
    }
    assert report["decision_summary"]["counts"] == {
        "AA": 1,
        "NN": 0,
        "AN": 0,
        "NA": 0,
    }
    assert report["quality_summary"]["correct_answer_rate"]["denominator"] == 1


def test_evaluate_cases_writes_json_after_every_response(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    snapshots = []
    original_save_results = evaluation.save_results

    def capture_save(report, path):
        snapshots.append(json.loads(json.dumps(report)))
        original_save_results(report, path)

    monkeypatch.setattr(
        evaluation,
        "build_metrics",
        lambda config: make_metrics(1.0, 1.0, 1.0, contextual=0.0),
    )
    monkeypatch.setattr(evaluation, "save_results", capture_save)

    asyncio.run(
        evaluate_cases(
            [make_case(True, True)],
            config={},
            max_workers=1,
            results_file=output,
        )
    )

    assert [
        snapshot["progress"]["metric_responses"] for snapshot in snapshots
    ] == [0, 1, 2, 3, 4]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["progress"]["status"] == "completed"
    assert saved["progress"]["completed_cases"] == 1
    assert not output.with_suffix(".json.tmp").exists()