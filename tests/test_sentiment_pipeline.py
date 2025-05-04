import pytest

from sentiment_analysis.sentiment_pipeline import SentimentAnalysisPipeline


@pytest.fixture
def sentiment_pipeline() -> SentimentAnalysisPipeline:
    """
    Fixture to create a SentimentAnalysisPipeline instance.
    """
    return SentimentAnalysisPipeline(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        label_mapping={"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0},
    )


@pytest.mark.parametrize(
    "text_input",
    [
        "I love this!",
        "This is awesome!",
        "I am so happy!",
        "This is the best day ever!",
        "I am thrilled with the results!",
    ],
)
def test_sentiment_analysis_pipeline_positive(sentiment_pipeline: SentimentAnalysisPipeline, text_input: str) -> None:
    """
    Test the sentiment analysis pipeline with a positive input.
    """
    assert sentiment_pipeline.run(text_input) > 0.0, "Expected positive sentiment score."


@pytest.mark.parametrize(
    "text_input",
    [
        "I hate this!",
        "This is terrible!",
        "I am so sad!",
        "This is the worst day ever!",
        "I am disappointed with the results!",
    ],
)
def test_sentiment_analysis_pipeline_negative(sentiment_pipeline: SentimentAnalysisPipeline, text_input: str) -> None:
    """
    Test the sentiment analysis pipeline with a negative input.
    """
    assert sentiment_pipeline.run(text_input) < 0.0, "Expected negative sentiment score."
