import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import load_models, load_employee_data, load_label_encoders, authenticate_employee, preprocess_img

class ImageAuthenticationApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Application d'Authentification")
        self.root.geometry("800x850")

        # Load your models, employee data, and label encoders
        self.left_eye_model, self.right_eye_model, self.od_og_model = load_models()
        self.employee_data = load_employee_data()
        self.labelEnc_left, self.labelEnc_right = load_label_encoders()

        self.select_button = tk.Button(root, text="Sélectionner une image", command=self.load_image)
        self.select_button.pack()
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        self.eye_image_label = tk.Label(root)
        self.eye_image_label.pack()

        self.authenticate_button = tk.Button(root, text="Authentifier", command=self.authenticate_employee)
        self.authenticate_button.pack()

        # Create a message frame
        self.message_frame = tk.Frame(root)
        self.message_frame.pack()

        # Add spacing between the "Authentifier" button and the message frame
        self.auth_message_space_label = tk.Label(self.message_frame, text="")
        self.auth_message_space_label.pack()

        self.authentication_label = tk.Label(self.message_frame, text="", justify="left", anchor="w", padx=10)

        # Add spacing between the message frame and the chart
        self.message_chart_space_label = tk.Label(root, text="")
        self.message_chart_space_label.pack()

        # Create a canvas for the bar chart
        self.chart_canvas = tk.Canvas(root, width=400, height=200)
        self.chart_canvas.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.imagepath_label.config(text=f"Image : {file_path}")
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
                self.eye_image_label.image = eye_photo
            else:
                self.eye_image_label.config(text="Erreur de prétraitement")
        else:
            self.eye_image_label.config(text="Aucune image sélectionnée")

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

            # Display the prediction in a bar chart
            if "Employé identifié" in authentication_result:
                self.display_bar_chart(authentication_result)
            else:
                self.chart_canvas.delete("all")  # Clear the canvas if no employee is identified

            # Update the message frame with the authentication result
            self.authentication_label.config(text=authentication_result)
            self.authentication_label.pack()  # Pack the label in the message frame

        else:
            self.authentication_label.config(text="Aucune image sélectionnée")
            self.authentication_label.pack()  # Pack the label in the message frame

    def display_bar_chart(self, authentication_result):
        # Parse the authentication result to extract employee ID and prediction probability
        parts = authentication_result.split(" ")
        employee_id = parts[2]
        prediction_probability = float(parts[-1])

        # Create a bar chart on the canvas
        self.chart_canvas.delete("all")  # Clear previous chart if any
        plt.figure(figsize=(4, 2))
        plt.bar(employee_id, prediction_probability, color='blue')
        plt.xlabel("Employee ID")
        plt.ylabel("Probabilité Prediction")
        plt.title("Employee ID vs. Prediction Probability")
        plt.tight_layout()

        # Save the bar chart to a file (you can adjust the filename as needed)
        chart_filename = "prediction_chart.png"
        plt.savefig(chart_filename)

        # Display the saved chart on the canvas
        chart_image = Image.open(chart_filename)
        chart_photo = ImageTk.PhotoImage(chart_image)
        self.chart_canvas.create_image(0, 0, anchor=tk.NW, image=chart_photo)
        self.chart_canvas.image = chart_photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAuthenticationApp(root)
    root.mainloop()
