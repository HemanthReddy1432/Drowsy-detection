import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

path = 'Training_images'

if not os.path.exists(path):
    os.makedirs(path)

def registerPerson(name):
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            messagebox.showerror("Error", "Could not read from the camera.")
            break
        cv2.imshow("Registration", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            return img, os.path.join(path, f"{name}.jpg")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None, None 

def register_with_gui():
    name = simpledialog.askstring("Input", "Enter the name of the person to register:")
    
    if name is not None:
        img, image_path = registerPerson(name)

        if img is not None:
            cv2.imwrite(image_path, img)
            messagebox.showinfo("Success", f"{name} registered successfully!\nImage saved as: {image_path}")
        else:
            messagebox.showinfo("Info", "Registration canceled.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  

    register_with_gui()
    root.quit()  
