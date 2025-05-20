import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")

with app.setup:
    import cv2
    import marimo as mo
    import numpy as np

    from sentiment_analysis.sentiment_pipeline import SentimentAnalysisPipeline
    from sentiment_analysis.utils import create_sentiment_image

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    label_mapping = {
        "LABEL_0": -1,  # Negative
        "LABEL_1": 0,  # Neutral
        "LABEL_2": 1,  # Positive
    }

    sentiment_analysis = SentimentAnalysisPipeline(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        label_mapping={"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1},
    )

    text_input = mo.ui.text(
        value="Love",
        # label="Text for sentiment analysis",
        placeholder="Enter text here...",
        full_width=True,
        debounce=10,
    )

    def process_text(text: str) -> tuple[np.ndarray, float]:
        """
        Process the input text and return the sentiment image and score.
        Args:
            text (str): The input text to analyze.
        Returns:
            tuple: A tuple containing the sentiment image and score.
            - sentiment_image (np.ndarray): The generated sentiment image.
            - positivity (float): The sentiment score.
        """
        positivity = sentiment_analysis.run(text)

        sentiment_image = create_sentiment_image(
            positivity=positivity,
            image_size=(1024, 1024),
        )

        sentiment_image = cv2.cvtColor(sentiment_image, cv2.COLOR_BGRA2RGBA)

        return sentiment_image, positivity


@app.cell
def process_text_cell() -> tuple[np.ndarray, float]:
    sentiment_image, positivity = process_text(text_input.value)
    return sentiment_image, positivity


@app.cell
def create_image_cell(sentiment_image: np.ndarray, positivity: float) -> tuple[mo.Html]:
    image_display = mo.lazy(
        mo.image(
            src=sentiment_image,
            alt="Sentiment visualization",
            width=256,
            height=256,
            rounded=True,
            caption=f"Score: {positivity:.2f}",
        )
    )
    return (image_display,)


@app.cell
def image_display_cell(image_display: mo.Html) -> None:
    mo.vstack(
        [
            mo.md("# Sentiment Analysis Visualization"),
            mo.callout(
                mo.vstack(
                    [
                        text_input,
                        image_display,
                    ],
                    align="center",
                    gap=1,
                )
            ),
        ],
        align="center",
        gap=1,
    )


@app.cell
def about_cell(model_name: str) -> None:
    mo.accordion(
        {
            "Instructions": mo.md(
                r"""
                1. Enter any text in the input field below.
                2. The sentiment score will be calculated and displayed.
                3. View the visual representation of the sentiment with a dynamic smiley face.
                """
            ),
            "About": mo.md(
                rf"""
                This tool uses a pre-trained sentiment analysis model from huggingface ({model_name}) to analyze the sentiment of the text you enter.
                The project is open-source and available on [GitHub](https://github.com/trflorian/sentiment-analysis-viz).
                """
            ),
        }
    )


if __name__ == "__main__":
    app.run()
