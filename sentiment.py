import logging
import cv2

import customtkinter
import numpy as np

from ctk_image_display import CTkImageDisplay


from transformers import pipeline


def color_hsv_to_bgr(
    hue: float, saturation: float, value: float
) -> tuple[int, int, int]:
    """
    Converts HSV color values to BGR format.

    Args:
        hue: Hue value in the range [0, 1].
        saturation: Saturation value in the range [0, 1].
        value: Value (brightness) in the range [0, 1].

    Returns:
        A tuple representing the BGR color.
    """
    bgr = cv2.cvtColor(
        np.uint8([[[hue * 180, saturation * 255, value * 255]]]), cv2.COLOR_HSV2BGR
    )
    return tuple(map(int, bgr[0][0]))  # Convert to tuple of integers


def create_sentiment_image(
    positivity: float, image_size: tuple[int, int]
) -> np.ndarray:
    """
    Generates a sentiment image based on the positivity score.
    This draws a smiley with its expression based on the positivity score.

    Args:
        positivity: A float representing the positivity score in the range [-1, 1].

    Returns:
        A string representing the path to the generated sentiment image.
    """
    frame = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    color_outline = (80,) * 3 + (255,)  # black
    thickness_outline = 40

    # draw a cricle in the center of the image which will be the head of the smiley
    center = (image_size[0] // 2, image_size[1] // 2)
    radius = min(image_size) // 2 - thickness_outline

    color = color_hsv_to_bgr(
        hue=(positivity + 1) / 6,
        saturation=0.5,
        value=1,
    )

    cv2.circle(frame, center, radius, color + (255,), -1)
    cv2.circle(frame, center, radius, color_outline, thickness_outline)

    # calculate the position of the eyes
    eye_radius = radius // 10
    eye_offset_x = radius // 3
    eye_offset_y = radius // 4
    eye_left = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    eye_right = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    cv2.circle(frame, eye_left, eye_radius, color_outline, -1)
    cv2.circle(frame, eye_right, eye_radius, color_outline, -1)

    # mouth parameters
    mouth_radius = radius // 2
    mouth_offset_y = radius // 4
    mouth_center_y = center[1] + mouth_offset_y + positivity * mouth_radius // 2
    mouth_left = (center[0] - mouth_radius, center[1] + mouth_offset_y)
    mouth_right = (center[0] + mouth_radius, center[1] + mouth_offset_y)

    # calculate points of polynomial for the mouth
    ply_points_t = np.linspace(-1, 1, 100)
    ply_points_y = np.polyval([positivity, 0, 0], ply_points_t)

    ply_points = np.array(
        [
            (
                mouth_left[0] + i * (mouth_right[0] - mouth_left[0]) / 100,
                mouth_center_y - ply_points_y[i] * mouth_radius,
            )
            for i in range(len(ply_points_y))
        ],
        dtype=np.int32,
    )

    # draw the mouth
    cv2.polylines(
        frame, [ply_points], isClosed=False, color=color_outline, thickness=int(thickness_outline * 1.5)
    )

    return frame


class App(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.sentiment_text_var.trace_add("write", self._on_text_change)

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

        self.image_display = CTkImageDisplay(
            self, display_size=(480, 480), canvas_size=(1024, 1024)
        )
        self.image_display.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        self._on_text_change()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _on_text_change(self, *args) -> None:
        """
        Callback function to handle text changes in the textbox.
        """
        new_text = self.sentiment_text_var.get()

        sentiment: list[dict] = self.sentiment_pipeline(new_text)[0]

        positivity = 0

        for label_score_dict in sentiment:
            label = label_score_dict["label"]
            score = label_score_dict["score"]

            if label in self.label_mapping:
                positivity += self.label_mapping[label] * score
                print(
                    f"Label: {label}, Score: {score}, Positivity: {positivity}"
                )
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


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    app = App()
    app.mainloop()
