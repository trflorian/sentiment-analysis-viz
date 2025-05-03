import logging

import customtkinter
import numpy as np
from transformers import pipeline

from sentiment_analysis.ctk_image_display import CTkImageDisplay
from sentiment_analysis.utils import create_sentiment_image


class App(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.title("Sentiment Analysis")
        self.geometry("800x600")

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=0,
            top_k=3,
        )
        self.label_mapping = {
            "LABEL_0": -1,
            "LABEL_1": 0,
            "LABEL_2": 1,
        }
        self.sentiment_image = None

        self.sentiment_text_var = customtkinter.StringVar(master=self, value="Love")
        self.sentiment_text_var.trace_add("write", lambda *_: self.update_sentiment())

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
        self.textbox.focus()

        self.image_display = CTkImageDisplay(self, display_size=(480, 480), canvas_size=(1024, 1024))
        self.image_display.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        self.update_sentiment()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_sentiment(self) -> None:
        """
        Callback function to handle text changes in the textbox.
        """
        new_text = self.sentiment_text_var.get()

        self.logger.info(f"New text: {new_text}")

        sentiment: list[dict] = self.sentiment_pipeline(new_text)[0]  # type: ignore

        positivity = 0

        for label_score_dict in sentiment:
            label = label_score_dict["label"]
            score = label_score_dict["score"]

            if label in self.label_mapping:
                positivity += self.label_mapping[label] * score

        positivity = np.clip(positivity, -1, 1)

        self.sentiment_image = create_sentiment_image(
            positivity,
            self.image_display.canvas_size,
        )

        self.image_display.update_frame(self.sentiment_image)

    def on_closing(self) -> None:
        """Handles window closing event."""
        self.logger.info("Closing application...")

        self.destroy()
        self.logger.info("Application closed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    app = App()
    app.mainloop()
