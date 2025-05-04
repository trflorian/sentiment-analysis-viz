import logging
import queue
import tkinter as tk
from typing import Any

import customtkinter
import cv2
import numpy.typing as npt
from PIL import Image


class CTkImageDisplay(customtkinter.CTkLabel):
    """
    A reusable CustomTkinter Label widget to display opencv images.
    """

    def __init__(
        self,
        master: Any,
        display_size: tuple[int, int] = (1024, 1024),
        refresh_dt_ms: int = 1000 // 60,
    ) -> None:
        self._textvariable = customtkinter.StringVar(master, "...")

        super().__init__(
            master,
            width=display_size[0],
            height=display_size[1],
            textvariable=self._textvariable,
            image=None,
        )

        self.logger = logging.getLogger(__name__)

        self.display_size = display_size
        self.widget_size = (
            self.winfo_width(),
            self.winfo_height(),
        )
        self.refresh_dt_ms = refresh_dt_ms

        self._frame = None
        self.frame_queue = queue.Queue[npt.NDArray](maxsize=2)
        self.bind("<Configure>", self._on_resize)

        self.after(self.refresh_dt_ms, self._on_refresh)

    def calculate_target_size(self, frame: npt.NDArray) -> tuple[int, int]:
        """
        Calculate the target size for the image based on the display size and aspect ratio.
        """
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        target_width, target_height = self.display_size if self.display_size else self.widget_size

        target_width = min(self.widget_size[0], target_width)
        target_height = min(self.widget_size[1], target_height)

        # make sure to keep the aspect ratio
        if frame_width / frame_height > target_width / target_height:
            target_height = int(frame_height * target_width / frame_width)
        else:
            target_width = int(frame_width * target_height / frame_height)

        return target_width, target_height

    def _set_frame(self, frame: npt.NDArray) -> None:
        """
        Set the frame to be displayed in the widget.
        This method is called when a new frame is available.

        Args:
            frame: The new frame to display, in BGR format.
        """

        self._frame = frame

        # Calculate the display size
        target_width, target_height = self.calculate_target_size(frame)

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

    def _on_refresh(self) -> None:
        """
        Handles the refresh event of the widget.
        This method is called periodically to update the displayed image.
        """
        # consume full queue
        frame = None
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._set_frame(frame)

        self.after(self.refresh_dt_ms, self._on_refresh)

    def _on_resize(self, event: tk.Event) -> None:
        """
        Handles the resize event of the widget.
        This method is called when the widget is resized.
        """
        self.widget_size = (
            event.width,
            event.height,
        )

        # update the displayed image if available
        if self._image is not None and self._frame is not None:
            self._set_frame(self._frame)

    def update_frame(self, frame: npt.NDArray) -> None:
        """
        Updates the displayed image with a new frame using CTkImage.

        Args:
            frame: The new frame to display, in BGR format.
        """
        self.frame_queue.put(frame)
