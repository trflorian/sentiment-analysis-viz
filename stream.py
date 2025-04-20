import threading
import cv2
from PIL import Image

import tkinter as tk
import customtkinter
import numpy as np

import queue


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
        resized_frame = cv2.resize(
            frame_bgr,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        ctk_image = customtkinter.CTkImage(
            light_image=frame_pil,
            dark_image=frame_pil,
            size=(self.display_width, self.display_height),
        )
        self.configure(image=ctk_image, text="")
        self._textvariable.set("")


# --- Constants ---
VIDEO_WIDTH = 400
VIDEO_HEIGHT = 240
QUEUE_CHECK_DELAY_MS = 30
MODE_NORMAL = 1
MODE_CANNY = 2


class App(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Threaded Webcam Stream")
        self.geometry("500x400")  # Adjusted size

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Image display row
        self.grid_rowconfigure(1, weight=0)  # Controls row

        self.radio_var = tk.IntVar(master=self, value=MODE_NORMAL)
        self._current_mode = MODE_NORMAL  # Store mode locally for processing check

        self.frame_queue = queue.LifoQueue(maxsize=2)
        self.stop_event = threading.Event()  # Signal for stopping the thread

        # --- Widgets ---
        self.image_display = CTkImageDisplay(
            self, width=VIDEO_WIDTH, height=VIDEO_HEIGHT
        )
        self.image_display.grid(
            row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        self.radiobutton_normal = customtkinter.CTkRadioButton(
            master=self,
            text="Normal",
            variable=self.radio_var,
            value=MODE_NORMAL,
            command=self._update_mode,  # Update local mode when radio changes
        )
        self.radiobutton_canny = customtkinter.CTkRadioButton(
            master=self,
            text="Canny Edges",
            variable=self.radio_var,
            value=MODE_CANNY,
            command=self._update_mode,  # Update local mode when radio changes
        )

        self.radiobutton_normal.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="e")
        self.radiobutton_canny.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="w")

        # --- Worker Thread ---
        self.video_thread = threading.Thread(
            target=self._video_capture_loop,
            args=(self.frame_queue, self.stop_event),
            daemon=True,
        )
        self.video_thread.start()

        # --- Start GUI Queue Checker ---
        self.after(QUEUE_CHECK_DELAY_MS, self.check_queue)

        # --- Shutdown Protocol ---
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _update_mode(self):
        """Callback when radio button changes to update local mode."""
        self._current_mode = self.radio_var.get()
        print(
            f"Mode changed to: {'Canny' if self._current_mode == MODE_CANNY else 'Normal'}"
        )

    def _video_capture_loop(self, frame_q: queue.Queue, stop_signal: threading.Event):
        """Function running in the worker thread to capture video frames."""
        print("Video thread started.")
        cap = cv2.VideoCapture(0)

        while not stop_signal.is_set():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera.")

            frame_q.put(frame)

        # --- Cleanup ---
        print("Video thread received stop signal. Releasing camera...")
        cap.release()
        print("Camera released in worker thread.")

    def check_queue(self):
        """Periodically checks the queue for new frames from the worker thread."""
        try:
            frame = self.frame_queue.get_nowait()

            if self._current_mode == MODE_CANNY:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_processed = cv2.Canny(frame_gray, 100, 200)
                frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)
            else:
                frame_processed = frame 

            self.image_display.update_frame(frame_processed)
        except queue.Empty:
            # No new frame available
            pass

        self.after(QUEUE_CHECK_DELAY_MS, self.check_queue)

    def on_closing(self) -> None:
        """Handles window closing event."""
        print("Closing application...")
        self.stop_event.set()

        print("Waiting for video thread to finish...")
        self.video_thread.join(timeout=1.0)  # Wait up to 1 second
        if self.video_thread.is_alive():
            print("Warning: Video thread did not finish cleanly.")

        self.destroy()
        print("Application closed.")


# --- Main Execution ---
if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    app = App()
    app.mainloop()
