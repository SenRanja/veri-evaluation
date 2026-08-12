import asyncio
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


def get_target_model(config):
    try:
        model = config["target"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain target.model") from error

    if not isinstance(model, str) or not model.strip():
        raise ValueError("target.model must be a non-empty string")
    return model.strip()


def load_cases(path, target_model):
    """Turn documents with multiple questions into individual test cases."""
    documents = load_json(path)
    cases = []
    answered_field = f"actual_answered_{target_model}"
    output_field = f"actual_output_{target_model}"

    for document in documents:
        for question in document["questions"]:
            if answered_field in question or output_field in question:
                actual_answered = question.get(answered_field)
                actual_output = question.get(output_field)
                selected_answered_field = answered_field
                selected_output_field = output_field
            else:
                actual_answered = question.get("actual_answered")
                actual_output = question.get("actual_output")
                selected_answered_field = "actual_answered"
                selected_output_field = "actual_output"

            if not isinstance(actual_answered, bool):
                raise ValueError(
                    f"{document['name']}/{question['name']} must contain "
                    f"Boolean {selected_answered_field}"
                )
            if not isinstance(actual_output, str) or not actual_output.strip():
                raise ValueError(
                    f"{document['name']}/{question['name']} must contain "
                    f"non-empty {selected_output_field}"
                )

            cases.append(
                {
                    **question,
                    "actual_answered": actual_answered,
                    "actual_output": actual_output,
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


def build_rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": safe_rate(numerator, denominator),
    }


def metric_gates_case(metric_name, actual_answered):
    if metric_name == "Correctness":
        return True
    if metric_name in {"Answer Relevancy", "Faithfulness"}:
        return actual_answered
    return False


def build_decision_summary(results):
    counts = {"AA": 0, "NN": 0, "AN": 0, "NA": 0}

    for result in results:
        counts[result["decision_state"]] += 1

    aa = counts["AA"]
    nn = counts["NN"]
    an = counts["AN"]
    na = counts["NA"]
    total = len(results)
    rate_details = {
        "decision_accuracy": build_rate(aa + nn, total),
        "false_refusal_rate": build_rate(an, aa + an),
        "hallucinated_answer_rate": build_rate(na, nn + na),
        "answer_precision": build_rate(aa, aa + na),
        "answer_recall": build_rate(aa, aa + an),
        "abstention_precision": build_rate(nn, nn + an),
    }

    return {
        "counts": counts,
        **{name: detail["value"] for name, detail in rate_details.items()},
        "rate_details": rate_details,
    }


def find_metric(result, metric_name):
    return next(
        metric for metric in result["metrics"]
        if metric["name"] == metric_name
    )


def build_metric_mean(results, decision_state, metric_name):
    scores = [
        find_metric(result, metric_name)["score"]
        for result in results
        if result["decision_state"] == decision_state
    ]
    return build_rate(sum(scores), len(scores))


def build_quality_summary(results):
    expected_answerable = [
        result for result in results if result["expected_answered"]
    ]
    correct_answers = sum(
        result["decision_state"] == "AA"
        and find_metric(result, "Correctness")["passed"]
        for result in expected_answerable
    )

    return {
        "aa_mean_correctness": build_metric_mean(
            results, "AA", "Correctness"
        ),
        "aa_mean_faithfulness": build_metric_mean(
            results, "AA", "Faithfulness"
        ),
        "aa_mean_answer_relevancy": build_metric_mean(
            results, "AA", "Answer Relevancy"
        ),
        "nn_mean_correctness": build_metric_mean(
            results, "NN", "Correctness"
        ),
        "correct_answer_rate": build_rate(
            correct_answers,
            len(expected_answerable),
        ),
    }


async def evaluate_case(case, metrics):
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    metric_results = []

    for metric in metrics:
        metric_name = getattr(
            metric,
            "name",
            metric.__class__.__name__.removesuffix("Metric"),
        )
        print(
            f"[{case['name']}] Evaluating {metric_name}...",
            flush=True,
        )
        await metric.a_measure(test_case, _show_indicator=False)

        score = float(metric.score)
        threshold = float(metric.threshold)

        metric_results.append(
            {
                "name": metric_name,
                "score": score,
                "threshold": threshold,
                "passed": score >= threshold,
                "gates_case": metric_gates_case(
                    metric_name,
                    case["actual_answered"],
                ),
                "reason": getattr(metric, "reason", None),
            }
        )

    decision_state = get_decision_state(case)
    decision_passed = decision_state in {"AA", "NN"}
    case_passed = decision_passed and all(
        result["passed"]
        for result in metric_results
        if result["gates_case"]
    )

    return {
        "document": case["document"],
        "name": case["name"],
        "expected_answered": case["expected_answered"],
        "actual_answered": case["actual_answered"],
        "decision_state": decision_state,
        "decision_passed": decision_passed,
        "case_passed": case_passed,
        "passed": case_passed,
        "input": case["input"],
        "actual_output": case["actual_output"],
        "expected_output": case["expected_output"],
        "retrieval_context": case["retrieval_context"],
        "metrics": metric_results,
    }


async def evaluate_cases(cases, config, max_workers):
    """Evaluate cases concurrently on one asyncio event loop."""
    semaphore = asyncio.Semaphore(max_workers)

    async def evaluate_with_limit(case):
        async with semaphore:
            metrics = build_metrics(config)
            result = await evaluate_case(case, metrics)
            print_case_result(result)
            return result

    return await asyncio.gather(
        *(evaluate_with_limit(case) for case in cases)
    )


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
        "metric_gates_case",
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
                        "metric_gates_case": metric["gates_case"],
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
        f"(decision={result['decision_state']})",
        flush=True,
    )

    for metric in result["metrics"]:
        metric_status = "PASS" if metric["passed"] else "FAIL"
        applicability = "GATE" if metric["gates_case"] else "DIAGNOSTIC"
        print(
            f"  {metric['name']}: {metric['score']:.4f} "
            f"[{metric_status}, {applicability}]"
        )


def main():
    config = load_yaml(CONFIG_FILE)
    project_root, run_directory = create_run_directory(config)

    cases_file = project_root / config["project"]["cases_file"]
    target_model = get_target_model(config)
    cases = load_cases(cases_file, target_model)
    max_workers = config.get("evaluation", {}).get("max_workers", 4)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("evaluation.max_workers must be an integer")
    if max_workers < 1:
        raise ValueError("evaluation.max_workers must be at least 1")

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

    print(
        f"Evaluating {len(cases)} cases with {max_workers} workers "
        f"for {target_model} using judge {config['judge']['model']}.",
        flush=True,
    )
    print(f"Output directory: {run_directory}", flush=True)

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

    with interceptor:
        results = asyncio.run(
            evaluate_cases(cases, config, max_workers)
        )

    decision_summary = build_decision_summary(results)
    quality_summary = build_quality_summary(results)

    save_results(
        {
            "decision_summary": decision_summary,
            "quality_summary": quality_summary,
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
    print(
        "Correct answer rate: "
        f"{quality_summary['correct_answer_rate']['value']}"
    )
    print(f"Output: {run_directory}")


if __name__ == "__main__":
    main()