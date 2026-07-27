import csv
import json
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


CASES_FILE = Path("evaluation_cases/test_cases.json")
RESULTS_DIR = Path("evaluation_results")
MODEL = "gpt-4o-mini"


def load_cases():
    with CASES_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def build_metrics():
    return [
        AnswerRelevancyMetric(
            threshold=0.7,
            model=MODEL,
        ),
        GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is correct "
                "based on the expected output."
            ),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
            model=MODEL,
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=MODEL,
        ),
    ]


def evaluate_case(case):
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    metric_results = []

    for metric in build_metrics():
        try:
            metric.measure(test_case)

            metric_results.append(
                {
                    "metric": metric.__class__.__name__,
                    "name": getattr(metric, "name", metric.__class__.__name__),
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "passed": metric.score >= metric.threshold,
                    "reason": metric.reason,
                    "error": None,
                }
            )
        except Exception as error:
            metric_results.append(
                {
                    "metric": metric.__class__.__name__,
                    "name": getattr(metric, "name", metric.__class__.__name__),
                    "score": None,
                    "threshold": metric.threshold,
                    "passed": False,
                    "reason": None,
                    "error": str(error),
                }
            )

    return {
        "name": case["name"],
        "input": case["input"],
        "actual_output": case["actual_output"],
        "expected_output": case["expected_output"],
        "retrieval_context": case["retrieval_context"],
        "passed": all(result["passed"] for result in metric_results),
        "metrics": metric_results,
    }


def save_json(results):
    output_file = RESULTS_DIR / "results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def save_csv(results):
    output_file = RESULTS_DIR / "summary.csv"

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "case",
                "metric",
                "score",
                "threshold",
                "passed",
                "reason",
                "error",
            ]
        )

        for result in results:
            for metric in result["metrics"]:
                writer.writerow(
                    [
                        result["name"],
                        metric["name"],
                        metric["score"],
                        metric["threshold"],
                        metric["passed"],
                        metric["reason"],
                        metric["error"],
                    ]
                )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    results = []

    for case in cases:
        print(f"Evaluating: {case['name']}")

        result = evaluate_case(case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"Result: {status}")

        for metric in result["metrics"]:
            print(
                f"  {metric['name']}: "
                f"{metric['score']} "
                f"(threshold={metric['threshold']})"
            )

    save_json(results)
    save_csv(results)

    print(f"\nJSON: {RESULTS_DIR / 'results.json'}")
    print(f"CSV:  {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()