import csv
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import yaml
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tools.openai_interceptor import OpenAIInterceptor


CONFIG_FILE = Path("config.yaml")


def load_yaml(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_cases(path):
    """Turn documents with multiple questions into individual test cases."""
    documents = load_json(path)
    cases = []

    for document in documents:
        for question in document["questions"]:
            cases.append(
                {
                    **question,
                    "document": document["name"],
                    "retrieval_context": document["retrieval_context"],
                }
            )

    return cases


def create_run_directory(config):
    project_root = CONFIG_FILE.resolve().parent
    results_root = project_root / config["project"]["results_directory"]
    model_name = config["judge"]["model"].replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_directory = results_root / f"{timestamp}-{model_name}"
    run_directory.mkdir(parents=True)

    return project_root, run_directory


def build_metrics(config):
    """Create the four metrics used in this evaluation."""
    model = config["judge"]["model"]
    settings = config["metrics"]

    contextual = settings.get("contextual_relevancy", {})
    answer = settings["answer_relevancy"]
    correctness = settings["correctness"]
    faithfulness = settings["faithfulness"]

    return [
        # Question <-> retrieved material
        ContextualRelevancyMetric(
            model=model,
            threshold=contextual.get("threshold", 0.5),
            include_reason=contextual.get("include_reason", True),
        ),

        # Question <-> model answer
        AnswerRelevancyMetric(
            model=model,
            threshold=answer["threshold"],
            include_reason=answer.get("include_reason", True),
        ),

        # Expected answer <-> model answer
        GEval(
            name="Correctness",
            model=model,
            criteria=correctness["criteria"],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=correctness["threshold"],
        ),

        # Retrieved material <-> model answer
        FaithfulnessMetric(
            model=model,
            threshold=faithfulness["threshold"],
            include_reason=faithfulness.get("include_reason", True),
        ),
    ]


def get_decision_state(case):
    expected = case["expected_answered"]
    actual = case["actual_answered"]

    if expected and actual:
        return "AA"
    if not expected and not actual:
        return "NN"
    if expected and not actual:
        return "AN"
    return "NA"


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def build_decision_summary(results):
    counts = {"AA": 0, "NN": 0, "AN": 0, "NA": 0}

    for result in results:
        counts[result["decision_state"]] += 1

    aa = counts["AA"]
    nn = counts["NN"]
    an = counts["AN"]
    na = counts["NA"]
    total = len(results)

    return {
        "counts": counts,
        "decision_accuracy": safe_rate(aa + nn, total),
        "false_refusal_rate": safe_rate(an, aa + an),
        "hallucinated_answer_rate": safe_rate(na, nn + na),
        "answer_precision": safe_rate(aa, aa + na),
        "answer_recall": safe_rate(aa, aa + an),
        "abstention_precision": safe_rate(nn, nn + an),
    }


def evaluate_case(case, metrics):
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    metric_results = []

    for metric in metrics:
        metric.measure(test_case)

        score = float(metric.score)
        threshold = float(metric.threshold)

        metric_results.append(
            {
                "name": getattr(
                    metric,
                    "name",
                    metric.__class__.__name__.removesuffix("Metric"),
                ),
                "score": score,
                "threshold": threshold,
                "passed": score >= threshold,
                "reason": getattr(metric, "reason", None),
            }
        )

    decision_state = get_decision_state(case)
    decision_passed = decision_state in {"AA", "NN"}

    return {
        "document": case["document"],
        "name": case["name"],
        "expected_answered": case["expected_answered"],
        "actual_answered": case["actual_answered"],
        "decision_state": decision_state,
        "decision_passed": decision_passed,
        "passed": (
            decision_passed
            and all(result["passed"] for result in metric_results)
        ),
        "input": case["input"],
        "actual_output": case["actual_output"],
        "expected_output": case["expected_output"],
        "retrieval_context": case["retrieval_context"],
        "metrics": metric_results,
    }


def save_results(results, output_file):
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def save_summary_csv(results, output_file):
    columns = [
        "document",
        "case",
        "expected_answered",
        "actual_answered",
        "decision_state",
        "decision_passed",
        "case_passed",
        "metric",
        "score",
        "threshold",
        "metric_passed",
        "reason",
    ]

    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for case in results:
            for metric in case["metrics"]:
                writer.writerow(
                    {
                        "document": case["document"],
                        "case": case["name"],
                        "expected_answered": case["expected_answered"],
                        "actual_answered": case["actual_answered"],
                        "decision_state": case["decision_state"],
                        "decision_passed": case["decision_passed"],
                        "case_passed": case["passed"],
                        "metric": metric["name"],
                        "score": metric["score"],
                        "threshold": metric["threshold"],
                        "metric_passed": metric["passed"],
                        "reason": metric["reason"],
                    }
                )


def build_token_summary(log_file):
    summary = {
        "response_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if not log_file.exists():
        return summary

    with log_file.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record.get("event") != "response":
                continue

            usage = record.get("usage") or {}
            prompt_tokens = usage.get(
                "prompt_tokens",
                usage.get("input_tokens", 0),
            )
            completion_tokens = usage.get(
                "completion_tokens",
                usage.get("output_tokens", 0),
            )

            summary["response_count"] += 1
            summary["prompt_tokens"] += prompt_tokens
            summary["completion_tokens"] += completion_tokens
            summary["total_tokens"] += usage.get(
                "total_tokens",
                prompt_tokens + completion_tokens,
            )

    return summary


def print_case_result(result):
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"{result['name']}: {status} "
        f"(decision={result['decision_state']})"
    )

    for metric in result["metrics"]:
        metric_status = "PASS" if metric["passed"] else "FAIL"
        print(
            f"  {metric['name']}: {metric['score']:.4f} "
            f"[{metric_status}]"
        )


def main():
    config = load_yaml(CONFIG_FILE)
    project_root, run_directory = create_run_directory(config)

    cases_file = project_root / config["project"]["cases_file"]
    cases = load_cases(cases_file)
    metrics = build_metrics(config)

    output = config["output"]
    interceptor_settings = config["openai_interceptor"]

    results_file = run_directory / output["results_json"]
    summary_file = run_directory / output["summary_csv"]
    token_file = run_directory / output["token_summary_json"]
    log_file = run_directory / interceptor_settings["log_file"]
    config_file = run_directory / output.get(
        "config_snapshot",
        "config_snapshot.yaml",
    )

    with config_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    interceptor = (
        OpenAIInterceptor(
            log_file=log_file,
            clear_existing=True,
            capture_full_messages=True,
            capture_full_response=True,
        )
        if interceptor_settings.get("enabled", True)
        else nullcontext()
    )

    results = []

    with interceptor:
        for case in cases:
            result = evaluate_case(case, metrics)
            results.append(result)
            print_case_result(result)

    decision_summary = build_decision_summary(results)

    save_results(
        {
            "decision_summary": decision_summary,
            "cases": results,
        },
        results_file,
    )
    save_summary_csv(results, summary_file)

    token_summary = build_token_summary(log_file)
    save_results(token_summary, token_file)

    passed = sum(result["passed"] for result in results)

    print("\nEvaluation summary")
    print(f"Cases:  {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"Tokens: {token_summary['total_tokens']}")
    print(f"AA:     {decision_summary['counts']['AA']}")
    print(f"NN:     {decision_summary['counts']['NN']}")
    print(f"AN:     {decision_summary['counts']['AN']}")
    print(f"NA:     {decision_summary['counts']['NA']}")
    print(
        "Decision accuracy: "
        f"{decision_summary['decision_accuracy']}"
    )
    print(f"Output: {run_directory}")


if __name__ == "__main__":
    main()