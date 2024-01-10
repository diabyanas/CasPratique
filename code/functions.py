import tensorflow as tf
import cv2
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder

def load_models():
    # Load your models here using your ML library
    left_eye_model = tf.keras.models.load_model("left_eye_model.h5")
    left_eye_model.trainable = False

    right_eye_model = tf.keras.models.load_model("right_eye_model.h5")
    right_eye_model.trainable = False

    od_og_model = tf.keras.models.load_model("OD_OG_Model.h5")
    od_og_model.trainable = False 

    return left_eye_model, right_eye_model, od_og_model

def load_employee_data():
    # Load the JSON file containing employee information
    with open("employees_info.json", "r") as json_file:
        employee_data = json.load(json_file)
    return employee_data

def load_label_encoders():
    # Load label encoders saved with joblib
    labelEnc_left = joblib.load('labelEnc_left.joblib')
    labelEnc_right = joblib.load('labelEnc_right.joblib')
    return labelEnc_left, labelEnc_right

def preprocess_img(img):
    # Preprocess the image as per the requirements of each model
    img = cv2.resize(img, (240, 320))
    img = img / 255.0  # Normalization
    return img

def authenticate_employee(image, od_og_model, labelEnc_left, labelEnc_right, employee_data, left_eye_model, right_eye_model):
    # Predict whether the eye is left or right
    eye_prediction = od_og_model.predict(np.array([image]))
    is_left_eye = eye_prediction[0][0] > eye_prediction[0][1]

    if is_left_eye:
        eye_model = left_eye_model
        label_encoder = labelEnc_left
    else:
        eye_model = right_eye_model
        label_encoder = labelEnc_right

    # Predict the eye label and get the prediction probability
    eye_prediction = eye_model.predict(np.array([image]))
    eye_label = label_encoder.inverse_transform([np.argmax(eye_prediction)])
    prediction_probability = np.max(eye_prediction)  # Get the maximum probability

    # Fetch employee information based on the predicted label
    if str(eye_label[0]) in employee_data.keys():
        employee_info = employee_data[str(eye_label[0])]
        message = f"Employé identifié: {str(eye_label[0])}\n"
        message += f"{employee_info['nom']} ({employee_info['genre']})\n"
        message += f"Poste: {employee_info['poste']}\n"
        message += f"Année d'Embauche: {employee_info['annee_embauche']}\n"
        message += f"Oeil Predit: {'Gauche' if is_left_eye else 'Droit'}\n"
        message += f"Probabilité Prediction: {prediction_probability:.2f}"
        return message
    else:
        return "Employee not identified"
