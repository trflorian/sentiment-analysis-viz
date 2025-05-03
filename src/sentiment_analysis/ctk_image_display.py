import logging
import tkinter as tk

import customtkinter
import cv2
import numpy as np
from PIL import Image


class CTkImageDisplay(customtkinter.CTkLabel):
    """
    A reusable CustomTkinter Label widget to display opencv images.
    """

    def __init__(
        self,
        master: customtkinter.CTk,
        display_size: tuple[int, int],
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self._textvariable = customtkinter.StringVar(master, "Loading...")
        super().__init__(
            master,
            width=display_size[0],
            height=display_size[1],
            textvariable=self._textvariable,
            image=None,
        )

        self.display_size = display_size
        self.canvas_size = canvas_size if canvas_size else display_size
        self.widget_size = (self.winfo_width(), self.winfo_height())

        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event: tk.Event) -> None:
        """
        Handles the resize event of the widget.
        This method is called when the widget is resized.
        """
        self.widget_size = (
            event.width,
            event.height,
        )
        self.logger.debug(f"Widget resized to: {self.widget_size}")
        self.update_frame(self._frame)

    def clear_frame(self) -> None:
        """
        Clears the displayed image and resets the text.
        """
        self.configure(image=None, text="")
        self._textvariable.set("")

    def update_frame(self, frame: np.ndarray) -> None:
        """
        Updates the displayed image with a new frame using CTkImage.

        Args:
            frame_bgr: The new frame to display, in BGR format.
        """
        self._frame = frame

        # Calculate the display size
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        target_width, target_height = self.display_size

        target_width = min(self.widget_size[0], target_width)
        target_height = min(self.widget_size[1], target_height)

        # make sure to keep the aspect ratio
        if frame_width > frame_height:
            target_height = int(frame_height * target_width / frame_width)
        else:
            target_width = int(frame_width * target_height / frame_height)
        
        # Resize the frame to fit the display size
        resized_frame = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        if resized_frame.shape[2] == 4:
            frame_rgba = cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2RGBA)
            frame_pil = Image.fromarray(frame_rgba, mode="RGBA")
        else:
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb, mode="RGB")

        ctk_image = customtkinter.CTkImage(
            light_image=frame_pil,
            dark_image=frame_pil,
            size=(target_width, target_height),
        )
        self.configure(image=ctk_image, text="")
        self._textvariable.set("")
