from types import SimpleNamespace

from judge_veri_answered import classify_response


class FakeResponses:
    def __init__(self, decision):
        self.decision = decision
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_parsed=SimpleNamespace(actual_answered=self.decision)
        )


def test_classify_response_returns_structured_judge_decision() -> None:
    responses = FakeResponses(False)
    client = SimpleNamespace(responses=responses)

    decision = classify_response(
        client,
        "gpt-4o-mini",
        "Who performed first?",
        "【Answer】\nNo relevant information was found.",
    )

    assert decision is False
    assert responses.request["model"] == "gpt-4o-mini"
    prompt = responses.request["input"][0]["content"]
    assert "Do not assess factual correctness" in prompt