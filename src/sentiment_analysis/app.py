from multiprocessing.pool import ThreadPool

import customtkinter

from sentiment_analysis.ctk_image_display import CTkImageDisplay
from sentiment_analysis.sentiment_pipeline import SentimentAnalysisPipeline
from sentiment_analysis.utils import create_sentiment_image


class App(customtkinter.CTk):
    def __init__(self, sentiment_analysis_pipeline: SentimentAnalysisPipeline) -> None:
        super().__init__()
        self.sentiment_analysis_pipeline = sentiment_analysis_pipeline

        self.title("Sentiment Analysis")
        self.geometry("800x600")

        self.sentiment_image = None

        self.sentiment_text_var = customtkinter.StringVar(master=self, value="Love")
        self.sentiment_text_var.trace_add("write", lambda *_: self.on_sentiment_text_changed())

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.textbox = customtkinter.CTkEntry(
            master=self,
            corner_radius=10,
            font=("Consolas", 50),
            justify="center",
            placeholder_text="Enter text here...",
            placeholder_text_color="gray",
            textvariable=self.sentiment_text_var,
        )
        self.textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.textbox.select_adjust(len(self.sentiment_text_var.get()))
        self.textbox.focus()

        self.image_display = CTkImageDisplay(self)
        self.image_display.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        self.update_sentiment_pool = ThreadPool(processes=1)

        self.on_sentiment_text_changed()

    def on_sentiment_text_changed(self) -> None:
        """
        Callback function to handle text changes in the textbox.
        """
        new_text = self.sentiment_text_var.get()

        self.update_sentiment_pool.apply_async(
            self._update_sentiment,
            (new_text,),
        )

    def _update_sentiment(self, new_text: str) -> None:
        """
        Update the sentiment image based on the new text input.
        This function is run in a separate process to avoid blocking the main thread.

        Args:
            new_text: The new text input from the user.
        """
        positivity = self.sentiment_analysis_pipeline.run(new_text)

        self.sentiment_image = create_sentiment_image(
            positivity,
            self.image_display.display_size,
        )

        self.image_display.update_frame(self.sentiment_image)


def main() -> None:
    # Initialize the sentiment analysis pipeline
    sentiment_analysis_config = SentimentAnalysisPipeline.Config(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        label_mapping={"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1},
    )
    sentiment_analysis = SentimentAnalysisPipeline(config=sentiment_analysis_config)

    app = App(sentiment_analysis)
    app.mainloop()


if __name__ == "__main__":
    main()
