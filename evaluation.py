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
METRIC_METADATA = {
    "ContextualRelevancyMetric": (
        "contextual_relevancy",
        "Contextual Relevancy",
    ),
    "AnswerRelevancyMetric": ("answer_relevancy", "Answer Relevancy"),
    "FaithfulnessMetric": ("faithfulness", "Faithfulness"),
}


def load_yaml(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def get_target_model(config):
    return get_target_models(config)[0]


def get_target_models(config):
    try:
        target = config["target"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain target") from error

    models = target.get("models")
    if models is None:
        models = [target.get("model")]
    if not isinstance(models, list) or not models:
        raise ValueError("target.models must be a non-empty list")
    if any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError("every target.models entry must be a non-empty string")

    normalized = [model.strip() for model in models]
    if len(set(normalized)) != len(normalized):
        raise ValueError("target.models must not contain duplicates")
    return normalized


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


def create_run_directory(config, target_model):
    project_root = CONFIG_FILE.resolve().parent
    results_root = project_root / config["project"]["results_directory"]
    target_name = target_model.replace("/", "-")
    judge_name = config["judge"]["model"].replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_directory = results_root / f"{timestamp}-{target_name}-judge-{judge_name}"
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


def get_metric_identity(metric):
    class_name = metric.__class__.__name__
    if class_name in METRIC_METADATA:
        return METRIC_METADATA[class_name]

    display_name = getattr(metric, "name", None) or class_name.removesuffix(
        "Metric"
    )
    return display_name.lower().replace(" ", "_"), display_name


def metric_gates_case(metric_id, actual_answered):
    if metric_id == "correctness":
        return True
    if metric_id in {"answer_relevancy", "faithfulness"}:
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


def find_metric(result, metric_id):
    return next(
        metric for metric in result["metrics"]
        if metric["id"] == metric_id
    )


def build_metric_mean(results, decision_state, metric_id):
    scores = [
        find_metric(result, metric_id)["score"]
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
        and find_metric(result, "correctness")["passed"]
        for result in expected_answerable
    )

    return {
        "aa_mean_correctness": build_metric_mean(
            results, "AA", "correctness"
        ),
        "aa_mean_faithfulness": build_metric_mean(
            results, "AA", "faithfulness"
        ),
        "aa_mean_answer_relevancy": build_metric_mean(
            results, "AA", "answer_relevancy"
        ),
        "nn_mean_correctness": build_metric_mean(
            results, "NN", "correctness"
        ),
        "correct_answer_rate": build_rate(
            correct_answers,
            len(expected_answerable),
        ),
    }


def new_case_result(case):
    decision_state = get_decision_state(case)
    return {
        "status": "in_progress",
        "document": case["document"],
        "name": case["name"],
        "expected_answered": case["expected_answered"],
        "actual_answered": case["actual_answered"],
        "decision_state": decision_state,
        "decision_passed": decision_state in {"AA", "NN"},
        "case_passed": None,
        "passed": None,
        "input": case["input"],
        "actual_output": case["actual_output"],
        "expected_output": case["expected_output"],
        "retrieval_context": case["retrieval_context"],
        "metrics": [],
    }


async def evaluate_case(case, metrics, on_metric_response=None):
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    result = new_case_result(case)

    for metric_index, metric in enumerate(metrics):
        metric_id, metric_name = get_metric_identity(metric)
        print(
            f"[{case['name']}] Evaluating {metric_name}...",
            flush=True,
        )
        await metric.a_measure(test_case, _show_indicator=False)

        score = float(metric.score)
        threshold = float(metric.threshold)

        result["metrics"].append(
            {
                "id": metric_id,
                "name": metric_name,
                "score": score,
                "threshold": threshold,
                "passed": score >= threshold,
                "gates_case": metric_gates_case(
                    metric_id,
                    case["actual_answered"],
                ),
                "reason": getattr(metric, "reason", None),
            }
        )

        if metric_index == len(metrics) - 1:
            case_passed = result["decision_passed"] and all(
                metric_result["passed"]
                for metric_result in result["metrics"]
                if metric_result["gates_case"]
            )
            result["status"] = "completed"
            result["case_passed"] = case_passed
            result["passed"] = case_passed

        if on_metric_response is not None:
            await on_metric_response(result)

    return result


def build_live_report(results, total_cases, metrics_per_case):
    available_results = [result for result in results if result is not None]
    completed_results = [
        result for result in available_results
        if result["status"] == "completed"
    ]
    metric_responses = sum(
        len(result["metrics"]) for result in available_results
    )

    return {
        "progress": {
            "status": (
                "completed"
                if len(completed_results) == total_cases
                else "running"
            ),
            "total_cases": total_cases,
            "completed_cases": len(completed_results),
            "metric_responses": metric_responses,
            "expected_metric_responses": total_cases * metrics_per_case,
        },
        "decision_summary": build_decision_summary(completed_results),
        "quality_summary": build_quality_summary(completed_results),
        "cases": available_results,
    }


async def evaluate_cases(cases, config, max_workers, results_file):
    """Evaluate cases concurrently on one asyncio event loop."""
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    results = [None] * len(cases)
    metrics_per_case = len(build_metrics(config))

    save_results(
        build_live_report(results, len(cases), metrics_per_case),
        results_file,
    )

    async def evaluate_with_limit(case_index, case):
        async with semaphore:
            metrics = build_metrics(config)

            async def save_metric_response(result):
                async with write_lock:
                    results[case_index] = result
                    report = build_live_report(
                        results,
                        len(cases),
                        metrics_per_case,
                    )
                    save_results(report, results_file)
                    if result["status"] == "completed":
                        print_case_result(result)
                        print_live_summary(report)

            result = await evaluate_case(
                case,
                metrics,
                on_metric_response=save_metric_response,
            )
            return result

    return await asyncio.gather(
        *(
            evaluate_with_limit(case_index, case)
            for case_index, case in enumerate(cases)
        )
    )


def save_results(results, output_file):
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(output_file)


def save_summary_csv(results, output_file):
    columns = [
        "document",
        "case",
        "expected_answered",
        "actual_answered",
        "decision_state",
        "decision_passed",
        "case_passed",
        "metric_id",
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
                        "metric_id": metric["id"],
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


def print_live_summary(report):
    progress = report["progress"]
    decision = report["decision_summary"]
    quality = report["quality_summary"]
    passed = sum(
        result["case_passed"] for result in report["cases"]
        if result["status"] == "completed"
    )
    print(
        "Live summary: "
        f"{progress['completed_cases']}/{progress['total_cases']} cases, "
        f"passed={passed}, "
        f"decision_accuracy={decision['decision_accuracy']}, "
        "correct_answer_rate="
        f"{quality['correct_answer_rate']['value']}",
        flush=True,
    )


def evaluate_target(config, project_root, target_model, max_workers):
    _, run_directory = create_run_directory(config, target_model)
    cases_file = project_root / config["project"]["cases_file"]
    cases = load_cases(cases_file, target_model)
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
        yaml.safe_dump(
            {**config, "active_target_model": target_model},
            file,
            allow_unicode=True,
            sort_keys=False,
        )

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
            evaluate_cases(cases, config, max_workers, results_file)
        )

    decision_summary = build_decision_summary(results)
    quality_summary = build_quality_summary(results)

    save_results(
        build_live_report(results, len(cases), len(build_metrics(config))),
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


def main():
    config = load_yaml(CONFIG_FILE)
    project_root = CONFIG_FILE.resolve().parent
    target_models = get_target_models(config)
    max_workers = config.get("evaluation", {}).get("max_workers", 4)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("evaluation.max_workers must be an integer")
    if max_workers < 1:
        raise ValueError("evaluation.max_workers must be at least 1")

    for target_model in target_models:
        evaluate_target(config, project_root, target_model, max_workers)


if __name__ == "__main__":
    main()