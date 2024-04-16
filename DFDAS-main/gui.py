import tkinter as tk
from tkinter import PhotoImage
import subprocess
from PIL import Image, ImageTk
import math


def authentication():
    subprocess.run(["python", "reg.py"])

def drowsiness():
    subprocess.run(["python", "integrated.py"])

def update_marquee():
    # Shift the text by one character to the right
    marquee_text.set(marquee_text.get()[1:] + marquee_text.get()[0])
    root.after(200, update_marquee)  # Repeat every 100 milliseconds


root = tk.Tk()
root.title("DDAS")
root.geometry("800x600")


bg_image = Image.open("3.jpg")  
bg_width, bg_height = 420, 555  
bg_image = bg_image.resize((bg_width, bg_height))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=190, y=29, relwidth=1, relheight=1)


custom_font = ("Verdana", 15, "bold")

marquee_text = tk.StringVar()
marquee_text.set("                    Drowsiness Detection and Alert System")


title_label = tk.Label(root, textvariable=marquee_text, font=("times new roman", 30, "bold"), bg="grey", fg="gold")
title_label.place(x=0, y=0, relwidth=1)


update_marquee()


button1 = tk.Button(root, text="Register", command=authentication, font=custom_font, bg="#3498db", fg="white", width=15, height=1)
button1.place(x=90, y=150)

button2 = tk.Button(root, text="Drowsiness Detection", command=drowsiness, font=custom_font, bg="#3498db", fg="white", width=20, height=1)
button2.place(x=50, y=250)


watermark_label = tk.Label(root, text="Capstone Project, Batch-10", font=("Verdana", 15, "italic"), fg="gray")
watermark_label.place(x=20, y=570)


def close_window():
    root.destroy()


exit_button = tk.Button(root, text="Exit", command=close_window, font=custom_font, bg="red", fg="white", width=10, height=1)
exit_button.place(x=120, y=350)

root.mainloop()
