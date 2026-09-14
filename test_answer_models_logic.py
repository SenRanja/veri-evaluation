import json
import threading
from types import SimpleNamespace

import answer_models
import pytest
from answer_models import (
    CandidateModel,
    ModelAnswer,
    build_tasks,
    execute_tasks,
    extract_source_citation,
    load_candidates,
    validate_answer,
)


def candidate(candidate_id: str) -> CandidateModel:
    return CandidateModel(
        id=candidate_id,
        model=f"{candidate_id}-api-model",
        api_key_env=f"{candidate_id.upper()}_API_KEY",
    )


def documents() -> list[dict]:
    return [
        {
            "name": "document",
            "retrieval_context": ["Alpha was founded in 1999."],
            "questions": [
                {
                    "name": "founded",
                    "input": "When was Alpha founded?",
                    "expected_answered": True,
                    "expected_output": "1999.",
                    "actual_answered": None,
                    "actual_output": None,
                },
                {
                    "name": "missing",
                    "input": "Where is Beta?",
                    "expected_answered": False,
                    "expected_output": "Not specified.",
                    "actual_answered": None,
                    "actual_output": None,
                },
            ],
        }
    ]


def test_load_candidates_supports_openai_and_deepseek() -> None:
    config = {
        "answering": {
            "models": [
                {
                    "id": "gpt-4o-mini",
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
                {
                    "id": "deepseek",
                    "model": "deepseek-flash",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/",
                },
            ]
        }
    }

    models = load_candidates(config)

    assert [model.id for model in models] == ["gpt-4o-mini", "deepseek"]
    assert models[1].model == "deepseek-flash"
    assert models[1].base_url == "https://api.deepseek.com"


def test_answer_validation_requires_source_citation_when_answered() -> None:
    output = 'Answer: 1999.\nSource citation: "Alpha was founded in 1999."'
    answer = ModelAnswer(actual_answered=True, actual_output=output)

    validate_answer(answer, "Alpha was founded in 1999.")

    assert extract_source_citation(output) == "Alpha was founded in 1999."


def test_answer_validation_allows_paraphrased_citation() -> None:
    answer = ModelAnswer(
        actual_answered=True,
        actual_output=(
            'Answer: 1999.\nSource citation: '
            '"The source says Alpha was established in 1999."'
        ),
    )

    validate_answer(answer, "Alpha was founded in 1999.")


def test_model_answer_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ModelAnswer.model_validate(
            {
                "actual_answered": False,
                "actual_output": "Not specified.",
                "score": 1,
            }
        )


def test_build_tasks_creates_model_specific_work_and_skips_complete() -> None:
    data = documents()
    data[0]["questions"][0]["actual_answered_gpt"] = True
    data[0]["questions"][0]["actual_output_gpt"] = "Existing."

    tasks, skipped = build_tasks(
        data, [candidate("gpt"), candidate("deepseek")], False, 0
    )

    assert skipped == 1
    assert [(task.question_index, task.candidate.id) for task in tasks] == [
        (0, "deepseek"),
        (1, "gpt"),
        (1, "deepseek"),
    ]


def test_concurrent_workers_only_commit_complete_pairs_on_main_thread(
    tmp_path, monkeypatch
) -> None:
    data = documents()
    models = [candidate("gpt"), candidate("deepseek")]
    tasks, _ = build_tasks(data, models, False, 0)
    output = tmp_path / "cases.json"
    output.write_text(json.dumps(data), encoding="utf-8")
    main_thread = threading.get_ident()
    save_threads = []
    original_save = answer_models.save_json

    def capture_save(payload, path):
        save_threads.append(threading.get_ident())
        original_save(payload, path)

    monkeypatch.setattr(answer_models, "save_json", capture_save)

    class FakeCompletions:
        def create(self, **kwargs):
            question = kwargs["messages"][1]["content"]
            if "When was Alpha" in question:
                content = json.dumps(
                    {
                        "actual_answered": True,
                        "actual_output": (
                            'Answer: 1999.\nSource citation: '
                            '"Alpha was founded in 1999."'
                        ),
                    }
                )
            else:
                content = json.dumps(
                    {
                        "actual_answered": False,
                        "actual_output": "The supplied context does not specify this.",
                    }
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    processed, errors = execute_tasks(
        data,
        tasks,
        "Prompt",
        retries=1,
        max_workers=4,
        output_path=output,
        client_factory=lambda _: fake_client,
    )

    assert processed == 4
    assert errors == []
    assert save_threads == [main_thread] * 4
    saved = json.loads(output.read_text(encoding="utf-8"))
    for question in saved[0]["questions"]:
        for model in models:
            assert isinstance(question[f"actual_answered_{model.id}"], bool)
            assert question[f"actual_output_{model.id}"]