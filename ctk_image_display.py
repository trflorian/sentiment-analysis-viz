import logging
from PIL import Image
import customtkinter
import cv2
import numpy as np


class CTkImageDisplay(customtkinter.CTkLabel):
    """
    A reusable CustomTkinter Label widget to display opencv images.
    """

    def __init__(self, master: any, width: int, height: int, *args, **kwargs):
        self._textvariable = customtkinter.StringVar(master, "Loading...")
        super().__init__(
            master,
            width=width,
            height=height,
            textvariable=self._textvariable,
            image=None,
            *args,
            **kwargs,
        )
        self.display_width = width
        self.display_height = height

        self.logger = logging.getLogger(__name__)

    def clear_frame(self) -> None:
        """
        Clears the displayed image and resets the text.
        """
        self.configure(image=None, text="")
        self._textvariable.set("")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Updates the displayed image with a new frame using CTkImage.

        Args:
            frame_bgr: The new frame to display, in BGR format.
        """

        ratio_frame = frame_bgr.shape[1] / frame_bgr.shape[0]
        ratio_target = self.display_width / self.display_height

        if ratio_frame != ratio_target:
            self.logger.warning(
                f"Aspect ratio mismatch: frame {ratio_frame:.2f} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]}) "
                f"vs target {ratio_target:.2f} ({self.display_width}x{self.display_height})"
            )

        if ratio_frame > 1:
            new_width = self.display_width
            new_height = int(self.display_width / ratio_frame)
        else:
            new_height = self.display_height
            new_width = int(self.display_height * ratio_frame)

        resized_frame = cv2.resize(
            frame_bgr,
            dsize=(new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        resized_frame = resized_frame[
            (resized_frame.shape[0] - self.display_height) // 2:
            (resized_frame.shape[0] + self.display_height) // 2,
            (resized_frame.shape[1] - self.display_width) // 2:
            (resized_frame.shape[1] + self.display_width) // 2,
        ]

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        ctk_image = customtkinter.CTkImage(
            light_image=frame_pil,
            dark_image=frame_pil,
            size=(self.display_width, self.display_height),
        )
        self.configure(image=ctk_image, text="")
        self._textvariable.set("")
