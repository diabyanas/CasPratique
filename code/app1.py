import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from functions import load_models, load_employee_data, load_label_encoders, preprocess_img, predict_eye_side, authenticate_employee

class ImageAuthenticationApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Authentication Application")
        self.root.geometry("800x600")

        # Load your models, employee data, and label encoders
        self.left_eye_model, self.right_eye_model, self.od_og_model = load_models()
        self.employee_data = load_employee_data()
        self.labelEnc_left, self.labelEnc_right = load_label_encoders()

        self.select_button = tk.Button(root, text="Select an Image", command=self.load_image)
        self.select_button.pack()
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        self.eye_image_label = tk.Label(root)  # Eye image display area
        self.eye_image_label.pack()
        self.authenticate_button = tk.Button(root, text="Authenticate", command=self.authenticate_employee)
        self.authenticate_button.pack()

        self.authentication_label = tk.Label(root, text="")
        self.authentication_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.imagepath_label.config(text=f"Image: {file_path}")
            self.load_eye_image()

    def load_eye_image(self):
        if hasattr(self, 'image'):
            # Preprocess the eye image using the preprocess_img function
            eye_image = preprocess_img(self.image)
            if eye_image is not None:
                # Convert the NumPy image to uint8 format
                eye_image = (eye_image * 255).astype(np.uint8)
                eye_photo = ImageTk.PhotoImage(Image.fromarray(eye_image))
                self.eye_image_label.config(image=eye_photo)
                self.eye_image_label.image = eye_photo  # Keep a reference to prevent image from being garbage collected
            else:
                self.eye_image_label.config(text="Preprocessing Error")
        else:
            self.eye_image_label.config(text="No image selected")

    def authenticate_employee(self):
        if hasattr(self, 'image'):
            authentication_result = authenticate_employee(
                self.image,
                self.od_og_model,
                self.labelEnc_left,
                self.labelEnc_right,
                self.employee_data,
                self.left_eye_model,
                self.right_eye_model
            )

            self.authentication_label.config(text=authentication_result)
        else:
            self.authentication_label.config(text="No image selected")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAuthenticationApp(root)
    root.mainloop()
