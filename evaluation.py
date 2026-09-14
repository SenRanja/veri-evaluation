import argparse
import asyncio
import csv
import json
import os
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
from deepeval.models import DeepSeekModel, OpenAIModel
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


def get_judge_models(config):
    try:
        judge = config["judge"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain judge") from error

    models = judge.get("models")
    if models is None:
        model = judge.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("judge.model must be a non-empty string")
        return [
            {
                "id": model.strip(),
                "provider": "openai",
                "model": model.strip(),
                "api_key_env": "OPENAI_API_KEY",
            }
        ]
    if not isinstance(models, list) or not models:
        raise ValueError("judge.models must be a non-empty list")

    normalized = []
    for index, item in enumerate(models, 1):
        if not isinstance(item, dict):
            raise ValueError(f"judge.models item {index} must be an object")
        values = {key: item.get(key) for key in ("id", "provider", "model")}
        if any(
            not isinstance(value, str) or not value.strip()
            for value in values.values()
        ):
            raise ValueError(
                f"judge.models item {index} requires id, provider and model"
            )
        provider = values["provider"].strip().lower()
        if provider not in {"openai", "deepseek"}:
            raise ValueError(
                f"judge.models item {index} has unsupported provider {provider}"
            )
        api_key_env = item.get(
            "api_key_env",
            "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY",
        )
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError(f"judge.models item {index} has invalid api_key_env")
        normalized.append(
            {
                "id": values["id"].strip(),
                "provider": provider,
                "model": values["model"].strip(),
                "api_key_env": api_key_env.strip(),
            }
        )

    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("judge.models IDs must not contain duplicates")
    return normalized


def create_judge_model(judge):
    api_key = os.environ.get(judge["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Missing environment variable {judge['api_key_env']} "
            f"for judge {judge['id']}"
        )
    if judge["provider"] == "deepseek":
        return DeepSeekModel(model=judge["model"], api_key=api_key)
    return OpenAIModel(model=judge["model"], api_key=api_key)


def load_cases(path, target_model, allow_partial=True):
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
            elif allow_partial:
                continue
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


def create_run_directory(config, target_model, judge_id=None):
    project_root = CONFIG_FILE.resolve().parent
    results_root = project_root / config["project"]["results_directory"]
    target_name = target_model.replace("/", "-")
    judge_name = (judge_id or config["judge"]["model"]).replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_directory = results_root / f"{timestamp}-{target_name}-judge-{judge_name}"
    run_directory.mkdir(parents=True)

    return project_root, run_directory


def build_metrics(config, judge_model=None):
    """Create the four metrics used in this evaluation."""
    model = judge_model or config["judge"]["model"]
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


def configured_metrics(config, judge_model=None):
    if judge_model is None:
        return build_metrics(config)
    return build_metrics(config, judge_model)


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


def format_metric_error(error):
    root_error = error
    seen = set()
    while id(root_error) not in seen:
        seen.add(id(root_error))
        nested = None
        last_attempt = getattr(root_error, "last_attempt", None)
        if last_attempt is not None:
            try:
                nested = last_attempt.exception()
            except Exception:
                nested = None
        nested = nested or root_error.__cause__ or root_error.__context__
        if not isinstance(nested, BaseException):
            break
        root_error = nested

    summary = f"{type(error).__name__}: {error}"
    if root_error is not error:
        summary += (
            f"; root cause: {type(root_error).__name__}: {root_error}"
        )
    return summary


async def evaluate_case(
    case,
    metrics,
    on_metric_response=None,
    metric_retries=3,
    existing_result=None,
):
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    result = (
        {
            **existing_result,
            "status": "in_progress",
            "case_passed": None,
            "passed": None,
            "metrics": list(existing_result["metrics"]),
        }
        if existing_result is not None
        else new_case_result(case)
    )
    result.pop("error", None)
    completed_metric_ids = [metric["id"] for metric in result["metrics"]]
    expected_metric_ids = [get_metric_identity(metric)[0] for metric in metrics]
    if completed_metric_ids != expected_metric_ids[:len(completed_metric_ids)]:
        raise ValueError(
            f"Cannot resume {case['document']}/{case['name']}: saved metrics "
            "are not a prefix of the configured metrics"
        )

    for metric_index, metric in enumerate(metrics):
        if metric_index < len(completed_metric_ids):
            continue
        metric_id, metric_name = get_metric_identity(metric)
        for attempt in range(1, metric_retries + 1):
            print(
                f"[{case['name']}] Evaluating {metric_name} "
                f"(attempt {attempt}/{metric_retries})...",
                flush=True,
            )
            try:
                await metric.a_measure(test_case, _show_indicator=False)
                break
            except Exception as error:
                error_summary = format_metric_error(error)
                if attempt == metric_retries:
                    result["status"] = "failed"
                    result["error"] = (
                        f"{metric_name} failed after {metric_retries} "
                        f"attempts: {error_summary}"
                    )
                    if on_metric_response is not None:
                        await on_metric_response(result)
                    return result
                print(
                    f"[{case['name']}] {metric_name} attempt {attempt} "
                    f"failed: {error_summary}; retrying.",
                    flush=True,
                )
                await asyncio.sleep(2 ** (attempt - 1))

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
    failed_results = [
        result for result in available_results
        if result["status"] == "failed"
    ]
    metric_responses = sum(
        len(result["metrics"]) for result in available_results
    )

    return {
        "progress": {
            "status": (
                "completed_with_errors"
                if failed_results and len(completed_results) + len(failed_results) == total_cases
                else "completed"
                if len(completed_results) == total_cases
                else "running"
            ),
            "total_cases": total_cases,
            "completed_cases": len(completed_results),
            "failed_cases": len(failed_results),
            "metric_responses": metric_responses,
            "expected_metric_responses": total_cases * metrics_per_case,
        },
        "decision_summary": build_decision_summary(completed_results),
        "quality_summary": build_quality_summary(completed_results),
        "cases": available_results,
    }


async def evaluate_cases(
    cases,
    config,
    max_workers,
    results_file,
    judge_model=None,
    existing_report=None,
    judge_id="unknown",
):
    """Evaluate cases concurrently on one asyncio event loop."""
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    expected_metric_ids = [
        get_metric_identity(metric)[0]
        for metric in configured_metrics(config, judge_model)
    ]
    metrics_per_case = len(expected_metric_ids)
    metric_retries = config.get("evaluation", {}).get("metric_retries", 3)
    results = (
        restore_saved_results(cases, existing_report, expected_metric_ids)
        if existing_report is not None
        else [None] * len(cases)
    )

    save_results(
        build_live_report(results, len(cases), metrics_per_case),
        results_file,
    )

    async def evaluate_with_limit(case_index, case):
        async with semaphore:
            metrics = configured_metrics(config, judge_model)

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
                    elif result["status"] == "failed":
                        print(
                            f"{result['name']}: EVALUATION ERROR "
                            f"(judge={judge_id}) - "
                            f"{result['error']}",
                            flush=True,
                        )
                        print_live_summary(report)

            result = await evaluate_case(
                case,
                metrics,
                on_metric_response=save_metric_response,
                metric_retries=metric_retries,
                existing_result=results[case_index],
            )
            return result

    pending = [
        (case_index, case)
        for case_index, case in enumerate(cases)
        if results[case_index] is None
        or results[case_index]["status"] != "completed"
    ]
    if existing_report is not None:
        print(
            f"Resume checkpoint: {len(cases) - len(pending)} completed, "
            f"{len(pending)} remaining.",
            flush=True,
        )

    await asyncio.gather(
        *(
            evaluate_with_limit(case_index, case)
            for case_index, case in pending
        )
    )
    return results


def restore_saved_results(cases, report, expected_metric_ids):
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise ValueError("Resume results.json must contain a cases list")
    progress = report.get("progress") or {}
    if progress.get("total_cases") != len(cases):
        raise ValueError(
            "Cannot resume because the checkpoint case count does not match "
            "the current input"
        )

    case_indexes = {}
    for index, case in enumerate(cases):
        key = (case["document"], case["name"])
        if key in case_indexes:
            raise ValueError(f"Duplicate current case identity: {key}")
        case_indexes[key] = index

    results = [None] * len(cases)
    comparable_fields = (
        "expected_answered",
        "actual_answered",
        "input",
        "actual_output",
        "expected_output",
        "retrieval_context",
    )
    for saved in report["cases"]:
        key = (saved.get("document"), saved.get("name"))
        if key not in case_indexes:
            raise ValueError(f"Resume checkpoint contains unknown case: {key}")
        index = case_indexes[key]
        if results[index] is not None:
            raise ValueError(f"Resume checkpoint contains duplicate case: {key}")
        current = new_case_result(cases[index])
        if any(saved.get(field) != current[field] for field in comparable_fields):
            raise ValueError(
                f"Cannot resume {key[0]}/{key[1]} because its input or answer "
                "data changed"
            )
        metrics = saved.get("metrics")
        if not isinstance(metrics, list) or len(metrics) > len(expected_metric_ids):
            raise ValueError(f"Resume checkpoint has invalid metrics for {key}")
        saved_metric_ids = [metric.get("id") for metric in metrics]
        if saved_metric_ids != expected_metric_ids[:len(saved_metric_ids)]:
            raise ValueError(
                f"Resume checkpoint metrics do not match the configuration: {key}"
            )
        if (
            saved.get("status") == "completed"
            and len(metrics) != len(expected_metric_ids)
        ):
            raise ValueError(f"Completed resume case has missing metrics: {key}")
        results[index] = saved
    return results


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
        f"failed={progress['failed_cases']}, "
        f"passed={passed}, "
        f"decision_accuracy={decision['decision_accuracy']}, "
        "correct_answer_rate="
        f"{quality['correct_answer_rate']['value']}",
        flush=True,
    )


def evaluate_target(
    config,
    project_root,
    target_model,
    judge,
    judge_model,
    max_workers,
    run_directory=None,
):
    cases_file = project_root / config["project"]["cases_file"]
    cases = load_cases(cases_file, target_model)
    if not cases:
        print(f"Skipping {target_model}: no complete target outputs.", flush=True)
        return
    is_resume = run_directory is not None
    if run_directory is None:
        _, run_directory = create_run_directory(config, target_model, judge["id"])
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

    if not is_resume:
        with config_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    **config,
                    "active_target_model": target_model,
                    "active_judge": judge,
                },
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    print(
        f"{'Resuming' if is_resume else 'Evaluating'} {len(cases)} cases "
        f"with {max_workers} workers "
        f"for {target_model} using judge {judge['id']}.",
        flush=True,
    )
    print(f"Output directory: {run_directory}", flush=True)

    interceptor = (
        OpenAIInterceptor(
            log_file=log_file,
            clear_existing=not is_resume,
            capture_full_messages=True,
            capture_full_response=True,
        )
        if interceptor_settings.get("enabled", True)
        else nullcontext()
    )

    with interceptor:
        results = asyncio.run(
            evaluate_cases(
                cases,
                config,
                max_workers,
                results_file,
                judge_model,
                load_json(results_file) if is_resume else None,
                judge["id"],
            )
        )

    completed_results = [
        result for result in results if result["status"] == "completed"
    ]
    failed_results = [
        result for result in results if result["status"] == "failed"
    ]
    decision_summary = build_decision_summary(completed_results)
    quality_summary = build_quality_summary(completed_results)

    save_results(
        build_live_report(
            results,
            len(cases),
            len(configured_metrics(config, judge_model)),
        ),
        results_file,
    )
    save_summary_csv(results, summary_file)

    token_summary = build_token_summary(log_file)
    save_results(token_summary, token_file)

    passed = sum(result["passed"] for result in completed_results)

    print("\nEvaluation summary")
    print(f"Cases:  {len(results)}")
    print(f"Evaluated: {len(completed_results)}")
    print(f"Evaluation errors: {len(failed_results)}")
    print(f"Passed: {passed}")
    print(f"Failed quality/decision: {len(completed_results) - passed}")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume one interrupted run directory in place.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override evaluation.max_workers for this invocation.",
    )
    return parser.parse_args()


def validate_max_workers(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("evaluation.max_workers must be an integer")
    if value < 1:
        raise ValueError("evaluation.max_workers must be at least 1")


def main():
    args = parse_args()
    config = load_yaml(CONFIG_FILE)
    project_root = CONFIG_FILE.resolve().parent
    if args.resume is not None:
        run_directory = args.resume
        if not run_directory.is_absolute():
            run_directory = project_root / run_directory
        run_directory = run_directory.resolve()
        config_file = run_directory / config["output"].get(
            "config_snapshot",
            "config_snapshot.yaml",
        )
        results_file = run_directory / config["output"]["results_json"]
        if not config_file.is_file() or not results_file.is_file():
            raise ValueError(
                "Resume directory must contain config_snapshot.yaml and "
                "results.json"
            )
        config = load_yaml(config_file)
        target_model = config.get("active_target_model")
        judge = config.get("active_judge")
        if not isinstance(target_model, str) or not isinstance(judge, dict):
            raise ValueError(
                "Resume config snapshot is missing active target or judge"
            )
        max_workers = args.max_workers or config.get("evaluation", {}).get(
            "max_workers",
            4,
        )
        validate_max_workers(max_workers)
        judge_model = create_judge_model(judge)
        evaluate_target(
            config,
            project_root,
            target_model,
            judge,
            judge_model,
            max_workers,
            run_directory=run_directory,
        )
        return

    target_models = get_target_models(config)
    judges = get_judge_models(config)
    max_workers = args.max_workers or config.get("evaluation", {}).get(
        "max_workers",
        4,
    )
    validate_max_workers(max_workers)
    metric_retries = config.get("evaluation", {}).get("metric_retries", 3)
    if not isinstance(metric_retries, int) or isinstance(metric_retries, bool):
        raise ValueError("evaluation.metric_retries must be an integer")
    if metric_retries < 1:
        raise ValueError("evaluation.metric_retries must be at least 1")

    for judge in judges:
        judge_model = create_judge_model(judge)
        for target_model in target_models:
            evaluate_target(
                config,
                project_root,
                target_model,
                judge,
                judge_model,
                max_workers,
            )


if __name__ == "__main__":
    main()