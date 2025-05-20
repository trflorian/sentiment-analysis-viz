import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from sentiment_analysis.sentiment_pipeline import SentimentAnalysisPipeline
    from sentiment_analysis.utils import create_sentiment_image

    sentiment_analysis = SentimentAnalysisPipeline(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        label_mapping={"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1},
    )

    text_input = mo.ui.text(
        value="Love",
        label="Text for sentiment analysis",
        placeholder="Enter text here...",
        full_width=True,
        debounce=10,
    )

    def create_image(texto):
        positivity = sentiment_analysis.run(texto)

        sentiment_image = create_sentiment_image(
                    positivity,
                    (224, 224),
                )
        return sentiment_image, positivity
    return create_image, mo, text_input


@app.cell
def _(create_image, text_input):
    sentiment_image, positivity = create_image(text_input.value)
    return positivity, sentiment_image


@app.cell
def _(mo, positivity, sentiment_image):
    # Create a color-coded sentiment score display
    sentiment_color = "green" if positivity > 0 else "red" if positivity < 0 else "gray"
    sentiment_text = "Positive" if positivity > 0 else "Negative" if positivity < 0 else "Neutral"

    # Create a styled sentiment score card
    score_card = mo.vstack([
        mo.md(f"## {sentiment_text}"),
        mo.md(f"### Score: {positivity:.2f}"),
    ], align="center", gap=1)

    image_display = mo.vstack([
        mo.image(
            src=sentiment_image,
            alt="Sentiment visualization",
            width=250,
            height=250,
            rounded=True,
            caption="Sentiment Results",
        ),
        score_card,
    ], align="center", gap=1)
    return (image_display,)


@app.cell
def _(mo):
    mo.md(
        """
    # 🎯 Sentiment Analysis Visualization

    Welcome to our interactive sentiment analysis tool!

    ### How it works:
    - Enter any text in the input field below
    - Get an instant sentiment analysis, view a visual representation of the sentiment and see a detailed score from -1 (negative) to 1 (positive)
    ---
    """
    )
    return


@app.cell
def _(mo, text_input):
    mo.vstack([
        mo.md("### ✍️ Enter your text"),
        text_input
    ], align="center", gap=1)
    return


@app.cell
def _(image_display, mo):
    mo.vstack([
        image_display
    ], align="center", gap=1)
    return


if __name__ == "__main__":
    app.run()
