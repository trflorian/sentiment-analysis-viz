import cv2
from PIL import Image, ImageTk

import tkinter as tk
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("600x400")


radio_var = tk.IntVar(master=app, value=1)


class StreamingCanvas(tk.Canvas):
    def __init__(self, master, cap, mode_var, *args, **kwargs):
        tk.Canvas.__init__(self, master, *args, **kwargs)
        self.cap = cap
        self.mode_var = mode_var
        self.update_image()

    def update_image(self):
        self.delete("all")

        ret, frame = self.cap.read()

        frame = cv2.resize(frame, (400, 240))
        if self.mode_var.get() == 2:
            frame = cv2.Canny(frame, 100, 200)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_cv2)

        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        # wait for the image to be displayed
        self.image = frame_tk
        self.create_image(0, 0, image=frame_tk, anchor=customtkinter.NW)

        self.after(10, self.update_image)


canvas = StreamingCanvas(
    app,
    cv2.VideoCapture(0),
    mode_var=radio_var,
    width=400,
    height=240,
    bd=0,
    highlightthickness=0,
)
canvas.place(relx=0.5, rely=0.4, anchor=customtkinter.CENTER)


def radiobutton_event():
    print("radiobutton toggled, current value:", radio_var.get())


radiobutton_1 = customtkinter.CTkRadioButton(
    master=app,
    text="Normal",
    command=radiobutton_event,
    variable=radio_var,
    value=1,
)
radiobutton_2 = customtkinter.CTkRadioButton(
    master=app,
    text="Canny",
    command=radiobutton_event,
    variable=radio_var,
    value=2,
)

# place radiobuttons
radiobutton_1.place(relx=0.3, rely=0.8, anchor=customtkinter.CENTER)
radiobutton_2.place(relx=0.7, rely=0.8, anchor=customtkinter.CENTER)

app.mainloop()
