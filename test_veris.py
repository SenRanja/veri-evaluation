from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_chatbot():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output=(
            "You have 30 days to return them "
            "for a full refund at no extra cost."
        ),
        expected_output=(
            "We offer a 30-day full refund at no extra costs."
        ),
        retrieval_context=[
            "All customers are eligible for a 30-day "
            "full refund at no extra costs."
        ],
    )

    metrics = [
        AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-5.4",
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
            model="gpt-5.4",
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model="gpt-5.4",
        ),
    ]

    assert_test(test_case, metrics)