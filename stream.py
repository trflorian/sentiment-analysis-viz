import logging
import threading
import cv2

import tkinter as tk
import customtkinter

import queue

from ctk_image_display import CTkImageDisplay


# --- Constants ---
QUEUE_CHECK_DELAY_MS = 30
MODE_NORMAL = 1
MODE_CANNY = 2


class App(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

        self.title("Webcam Stream")
        self.geometry("800x600")

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.radio_var = tk.IntVar(master=self, value=MODE_NORMAL)
        self._current_mode = MODE_NORMAL

        self.frame_queue = queue.LifoQueue(maxsize=2)
        self.stop_event = threading.Event()

        self.image_display = CTkImageDisplay(self, width=640, height=480)
        self.image_display.grid(
            row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        self.radiobutton_normal = customtkinter.CTkRadioButton(
            master=self,
            text="Normal",
            variable=self.radio_var,
            value=MODE_NORMAL,
            command=self._update_mode,
        )
        self.radiobutton_canny = customtkinter.CTkRadioButton(
            master=self,
            text="Canny Edges",
            variable=self.radio_var,
            value=MODE_CANNY,
            command=self._update_mode,
        )

        self.radiobutton_normal.grid(row=1, column=0, padx=(10, 5), pady=20, sticky="e")
        self.radiobutton_canny.grid(row=1, column=1, padx=(5, 10), pady=20, sticky="w")

        self.video_thread = threading.Thread(
            target=self._video_capture_loop,
            args=(self.frame_queue, self.stop_event),
            daemon=True,
        )
        self.video_thread.start()

        self.check_queue()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _update_mode(self):
        """Callback when radio button changes to update local mode."""
        self._current_mode = self.radio_var.get()
        self.logger.info(
            f"Mode changed to: {'Canny' if self._current_mode == MODE_CANNY else 'Normal'}"
        )

    def _video_capture_loop(self, frame_q: queue.Queue, stop_signal: threading.Event):
        """Function running in the worker thread to capture video frames."""
        self.logger.info("Video thread started.")
        cap = cv2.VideoCapture(0)

        while not stop_signal.is_set():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera.")

            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                # Queue is full, drop the oldest frame
                pass

        # --- Cleanup ---
        self.logger.info("Video thread received stop signal. Releasing camera...")
        cap.release()
        self.logger.info("Camera released in worker thread.")

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
        self.logger.info("Closing application...")
        self.stop_event.set()

        self.logger.info("Waiting for video thread to finish...")
        self.video_thread.join(timeout=0.1)
        if self.video_thread.is_alive():
            self.logger.warning("Warning: Video thread did not finish cleanly.")

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
