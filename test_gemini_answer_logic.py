import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).with_name("genimi-3.5-flash_answer.py")
SPEC = importlib.util.spec_from_file_location("gemini_answer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
gemini_answer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemini_answer)


class FakeInteractions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_text=(
                '{"actual_answered":true,'
                '"actual_output":"Wireless communications."}'
            )
        )


def test_ask_question_uses_context_and_structured_output() -> None:
    interactions = FakeInteractions()
    client = SimpleNamespace(interactions=interactions)

    answer = gemini_answer.ask_question(
        client,
        "gemini-3.5-flash",
        "Ahmad Bazzi specializes in wireless communications.",
        "What field does Ahmad Bazzi specialize in?",
    )

    assert answer.actual_answered is True
    assert answer.actual_output == "Wireless communications."
    assert interactions.request["model"] == "gemini-3.5-flash"
    assert "wireless communications" in interactions.request["input"]
    assert interactions.request["response_format"]["mime_type"] == "application/json"
    schema = interactions.request["response_format"]["schema"]
    assert set(schema["required"]) == {"actual_answered", "actual_output"}