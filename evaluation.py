from __future__ import annotations

import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from datetime import datetime

import re
import yaml
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tools.openai_interceptor import OpenAIInterceptor


CONFIG_FILE = Path("config.yaml")

def sanitize_path_component(value: str) -> str:
    """
    Convert text into a safe Windows directory-name component.

    Example:
        openai/gpt-4o-mini -> openai-gpt-4o-mini
    """
    sanitized = re.sub(
        r'[<>:"/\\|?*]+',
        "-",
        str(value).strip(),
    )

    sanitized = re.sub(r"\s+", "-", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)

    return sanitized.strip(" .-") or "unknown"


def build_run_name(config: dict[str, Any]) -> str:
    """
    Build one unique directory name for the current evaluation run.

    Example:
        20260727-224215-gpt-4o-mini
    """
    timestamp_format = config["output"].get(
        "timestamp_format",
        "%Y%m%d-%H%M%S",
    )

    timestamp = datetime.now().strftime(timestamp_format)

    include_model = bool(
        config["output"].get(
            "include_model_in_run_directory",
            True,
        )
    )

    if not include_model:
        return timestamp

    model = sanitize_path_component(
        config["judge"]["model"]
    )

    return f"{timestamp}-{model}"

class ConfigurationError(ValueError):
    """Raised when config.yaml is missing or invalid."""

def load_config(config_file: Path) -> dict[str, Any]:
    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_file.resolve()}"
        )

    with config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigurationError(
            "config.yaml must contain a YAML mapping."
        )

    required_sections = [
        "project",
        "judge",
        "metrics",
        "output",
        "openai_interceptor",
        "execution",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ConfigurationError(
            "Missing configuration sections: "
            + ", ".join(missing_sections)
        )

    return config


def build_run_timestamp() -> str:
    """
    Example:
        20260727224215
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")

def resolve_project_paths(
    config: dict[str, Any],
) -> dict[str, Path]:
    """
    Resolve all configured paths relative to config.yaml.

    Each evaluation run receives its own directory, for example:

        evaluation_results/
            20260727-224215-gpt-4o-mini/
                results.json
                summary.csv
                token_summary.json
                openai_interactions.jsonl
                config_snapshot.yaml
    """
    project_root = CONFIG_FILE.resolve().parent

    project_config = config["project"]
    output_config = config["output"]
    interceptor_config = config["openai_interceptor"]

    results_root_directory = (
        project_root
        / project_config["results_directory"]
    ).resolve()

    run_name = build_run_name(config)

    run_directory = (
        results_root_directory / run_name
    ).resolve()

    return {
        "project_root": project_root,
        "cases_file": (
            project_root / project_config["cases_file"]
        ).resolve(),

        # Parent directory containing every evaluation run.
        "results_root_directory": results_root_directory,

        # Directory belonging only to the current run.
        "results_directory": run_directory,
        "run_directory": run_directory,
        "run_name": Path(run_name),

        "results_json": (
            run_directory / output_config["results_json"]
        ).resolve(),

        "summary_csv": (
            run_directory / output_config["summary_csv"]
        ).resolve(),

        "token_summary_json": (
            run_directory
            / output_config["token_summary_json"]
        ).resolve(),

        "openai_log": (
            run_directory / interceptor_config["log_file"]
        ).resolve(),

        "config_snapshot": (
            run_directory
            / output_config.get(
                "config_snapshot",
                "config_snapshot.yaml",
            )
        ).resolve(),
    }


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    if not cases_file.exists():
        raise FileNotFoundError(
            f"Test case file not found: {cases_file}"
        )

    with cases_file.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError(
            "The test case JSON root must be an array."
        )

    required_fields = {
        "name",
        "input",
        "actual_output",
        "expected_output",
        "retrieval_context",
    }

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"Test case at index {index} must be an object."
            )

        missing_fields = required_fields - case.keys()

        if missing_fields:
            raise ValueError(
                f"Test case at index {index} is missing: "
                f"{', '.join(sorted(missing_fields))}"
            )

        if not isinstance(case["retrieval_context"], list):
            raise ValueError(
                f"Test case '{case['name']}' must use a list "
                "for retrieval_context."
            )

    return cases


def parse_single_turn_params(
    parameter_names: list[str],
) -> list[SingleTurnParams]:
    supported_params = {
        "INPUT": SingleTurnParams.INPUT,
        "ACTUAL_OUTPUT": SingleTurnParams.ACTUAL_OUTPUT,
        "EXPECTED_OUTPUT": SingleTurnParams.EXPECTED_OUTPUT,
        "RETRIEVAL_CONTEXT": (
            SingleTurnParams.RETRIEVAL_CONTEXT
        ),
        "CONTEXT": SingleTurnParams.CONTEXT,
    }

    parsed_params: list[SingleTurnParams] = []

    for name in parameter_names:
        normalized_name = str(name).strip().upper()

        if normalized_name not in supported_params:
            raise ConfigurationError(
                f"Unsupported GEval evaluation parameter: {name}. "
                f"Supported values: "
                f"{', '.join(supported_params)}"
            )

        parsed_params.append(supported_params[normalized_name])

    return parsed_params


def build_metrics(
    config: dict[str, Any],
) -> list[Any]:
    model = config["judge"]["model"]
    metrics_config = config["metrics"]

    metrics: list[Any] = []

    answer_config = metrics_config["answer_relevancy"]

    if answer_config.get("enabled", True):
        metrics.append(
            AnswerRelevancyMetric(
                threshold=float(answer_config["threshold"]),
                model=model,
                include_reason=bool(
                    answer_config.get("include_reason", True)
                ),
                strict_mode=bool(
                    answer_config.get("strict_mode", False)
                ),
                async_mode=bool(
                    answer_config.get("async_mode", True)
                ),
                verbose_mode=bool(
                    answer_config.get("verbose_mode", False)
                ),
            )
        )

    correctness_config = metrics_config["correctness"]

    if correctness_config.get("enabled", True):
        metrics.append(
            GEval(
                name=correctness_config.get(
                    "name",
                    "Correctness",
                ),
                criteria=correctness_config["criteria"],
                evaluation_params=parse_single_turn_params(
                    correctness_config["evaluation_params"]
                ),
                threshold=float(
                    correctness_config["threshold"]
                ),
                model=model,
                strict_mode=bool(
                    correctness_config.get(
                        "strict_mode",
                        False,
                    )
                ),
                async_mode=bool(
                    correctness_config.get(
                        "async_mode",
                        True,
                    )
                ),
                verbose_mode=bool(
                    correctness_config.get(
                        "verbose_mode",
                        False,
                    )
                ),
            )
        )

    faithfulness_config = metrics_config["faithfulness"]

    if faithfulness_config.get("enabled", True):
        metrics.append(
            FaithfulnessMetric(
                threshold=float(
                    faithfulness_config["threshold"]
                ),
                model=model,
                include_reason=bool(
                    faithfulness_config.get(
                        "include_reason",
                        True,
                    )
                ),
                strict_mode=bool(
                    faithfulness_config.get(
                        "strict_mode",
                        False,
                    )
                ),
                async_mode=bool(
                    faithfulness_config.get(
                        "async_mode",
                        True,
                    )
                ),
                verbose_mode=bool(
                    faithfulness_config.get(
                        "verbose_mode",
                        False,
                    )
                ),
                penalize_ambiguous_claims=bool(
                    faithfulness_config.get(
                        "penalize_ambiguous_claims",
                        False,
                    )
                ),
            )
        )

    if not metrics:
        raise ConfigurationError(
            "At least one metric must be enabled."
        )

    return metrics


def metric_display_name(metric: Any) -> str:
    return str(
        getattr(
            metric,
            "name",
            metric.__class__.__name__,
        )
    )


def evaluate_case(
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )

    metric_results: list[dict[str, Any]] = []
    continue_on_error = bool(
        config["execution"].get(
            "continue_on_metric_error",
            True,
        )
    )

    for metric in build_metrics(config):
        try:
            metric.measure(test_case)

            score = getattr(metric, "score", None)
            threshold = getattr(metric, "threshold", None)

            passed = (
                score is not None
                and threshold is not None
                and float(score) >= float(threshold)
            )

            metric_results.append(
                {
                    "metric_class": metric.__class__.__name__,
                    "name": metric_display_name(metric),
                    "score": score,
                    "threshold": threshold,
                    "passed": passed,
                    "reason": getattr(metric, "reason", None),
                    "error": None,
                }
            )

        except Exception as error:
            metric_results.append(
                {
                    "metric_class": metric.__class__.__name__,
                    "name": metric_display_name(metric),
                    "score": None,
                    "threshold": getattr(
                        metric,
                        "threshold",
                        None,
                    ),
                    "passed": False,
                    "reason": None,
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

            if not continue_on_error:
                raise

    result: dict[str, Any] = {
        "name": case["name"],
        "passed": all(
            metric_result["passed"]
            for metric_result in metric_results
        ),
        "metrics": metric_results,
    }

    if config["output"].get(
        "include_test_case_content",
        True,
    ):
        result.update(
            {
                "input": case["input"],
                "actual_output": case["actual_output"],
                "expected_output": case["expected_output"],
                "retrieval_context": case[
                    "retrieval_context"
                ],
            }
        )

    return result


def save_results_json(
    results: list[dict[str, Any]],
    output_file: Path,
    config: dict[str, Any],
) -> None:
    indent = (
        2
        if config["output"].get("pretty_json", True)
        else None
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=indent,
        )


def save_summary_csv(
    results: list[dict[str, Any]],
    output_file: Path,
) -> None:
    fieldnames = [
        "case",
        "case_passed",
        "metric",
        "metric_class",
        "score",
        "threshold",
        "metric_passed",
        "reason",
        "error",
    ]

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in results:
            for metric in result["metrics"]:
                writer.writerow(
                    {
                        "case": result["name"],
                        "case_passed": result["passed"],
                        "metric": metric["name"],
                        "metric_class": metric[
                            "metric_class"
                        ],
                        "score": metric["score"],
                        "threshold": metric["threshold"],
                        "metric_passed": metric["passed"],
                        "reason": metric["reason"],
                        "error": metric["error"],
                    }
                )


def build_token_summary(
    log_file: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "request_count": 0,
        "response_count": 0,
        "error_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests_by_model": {},
    }

    if not log_file.exists():
        summary["warning"] = (
            "OpenAI interaction log was not created. "
            "No intercepted Chat Completions calls were detected."
        )
        return summary

    with log_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                summary.setdefault(
                    "invalid_log_lines",
                    [],
                ).append(line_number)
                continue

            event = record.get("event")

            if event == "request":
                summary["request_count"] += 1

            elif event == "error":
                summary["error_count"] += 1

            elif event == "response":
                summary["response_count"] += 1

                usage = record.get("usage") or {}

                prompt_tokens = (
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or 0
                )
                completion_tokens = (
                    usage.get("completion_tokens")
                    or usage.get("output_tokens")
                    or 0
                )
                total_tokens = usage.get("total_tokens") or (
                    prompt_tokens + completion_tokens
                )

                summary["prompt_tokens"] += prompt_tokens
                summary["completion_tokens"] += (
                    completion_tokens
                )
                summary["total_tokens"] += total_tokens

                model = record.get("model") or "unknown"

                model_summary = summary[
                    "requests_by_model"
                ].setdefault(
                    model,
                    {
                        "response_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )

                model_summary["response_count"] += 1
                model_summary["prompt_tokens"] += (
                    prompt_tokens
                )
                model_summary["completion_tokens"] += (
                    completion_tokens
                )
                model_summary["total_tokens"] += total_tokens

    return summary

def save_config_snapshot(
    config: dict[str, Any],
    output_file: Path,
) -> None:
    """
    Save the exact configuration used by this evaluation run.
    """
    with output_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

def save_token_summary(
    token_summary: dict[str, Any],
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            token_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_case_result(
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"Result: {status}")

    print_reasons = bool(
        config["execution"].get(
            "print_metric_reasons",
            False,
        )
    )

    for metric in result["metrics"]:
        score = (
            f"{metric['score']:.4f}"
            if isinstance(metric["score"], (int, float))
            else "N/A"
        )

        metric_status = (
            "PASS" if metric["passed"] else "FAIL"
        )

        print(
            f"  [{metric_status}] {metric['name']}: "
            f"{score} "
            f"(threshold={metric['threshold']})"
        )

        if metric["error"]:
            print(f"    Error: {metric['error']}")

        elif print_reasons and metric["reason"]:
            print(f"    Reason: {metric['reason']}")


def print_run_summary(
    results: list[dict[str, Any]],
    token_summary: dict[str, Any],
) -> None:
    total_cases = len(results)
    passed_cases = sum(
        1 for result in results if result["passed"]
    )
    failed_cases = total_cases - passed_cases

    print("\nEvaluation summary")
    print("------------------")
    print(f"Cases:             {total_cases}")
    print(f"Passed:            {passed_cases}")
    print(f"Failed:            {failed_cases}")
    print(
        f"OpenAI responses:  "
        f"{token_summary['response_count']}"
    )
    print(
        f"Input tokens:      "
        f"{token_summary['prompt_tokens']}"
    )
    print(
        f"Output tokens:     "
        f"{token_summary['completion_tokens']}"
    )
    print(
        f"Total tokens:      "
        f"{token_summary['total_tokens']}"
    )


def create_interceptor_context(
    config: dict[str, Any],
    paths: dict[str, Path],
):
    interceptor_config = config["openai_interceptor"]

    if not interceptor_config.get("enabled", True):
        return nullcontext()

    return OpenAIInterceptor(
        log_file=paths["openai_log"],
        clear_existing=bool(
            interceptor_config.get(
                "clear_existing_log",
                True,
            )
        ),
        print_requests=bool(
            interceptor_config.get(
                "print_requests",
                False,
            )
        ),
        print_responses=bool(
            interceptor_config.get(
                "print_responses",
                False,
            )
        ),
        capture_full_messages=bool(
            interceptor_config.get(
                "capture_full_messages",
                True,
            )
        ),
        capture_full_response=bool(
            interceptor_config.get(
                "capture_full_response",
                True,
            )
        ),
    )


def main() -> int:
    try:
        config = load_config(CONFIG_FILE)
        paths = resolve_project_paths(config)

        paths["results_directory"].mkdir(
            parents=True,
            exist_ok=False,
        )

        save_config_snapshot(
            config,
            paths["config_snapshot"],
        )

        cases = load_cases(paths["cases_file"])

        results: list[dict[str, Any]] = []

        interceptor_context = create_interceptor_context(
            config,
            paths,
        )

        with interceptor_context:
            for case in cases:
                print(f"\nEvaluating: {case['name']}")

                result = evaluate_case(case, config)
                results.append(result)

                print_case_result(result, config)

        save_results_json(
            results,
            paths["results_json"],
            config,
        )
        save_summary_csv(
            results,
            paths["summary_csv"],
        )

        token_summary = build_token_summary(
            paths["openai_log"]
        )
        save_token_summary(
            token_summary,
            paths["token_summary_json"],
        )

        print_run_summary(results, token_summary)

        print("\nGenerated files")
        print("---------------")
        print(f"Results JSON:  {paths['results_json']}")
        print(f"Summary CSV:   {paths['summary_csv']}")
        print(
            f"Token summary: "
            f"{paths['token_summary_json']}"
        )

        if config["openai_interceptor"].get(
            "enabled",
            True,
        ):
            print(f"OpenAI log:    {paths['openai_log']}")

        # The report generator ran successfully even when cases failed.
        return 0

    except (
        ConfigurationError,
        FileNotFoundError,
        json.JSONDecodeError,
        yaml.YAMLError,
        ValueError,
    ) as error:
        print(
            f"Configuration or input error: {error}",
            file=sys.stderr,
        )
        return 2

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.", file=sys.stderr)
        return 130

    except Exception as error:
        print(
            f"Unexpected evaluation error: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())