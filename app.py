import flask
from flask import Flask, request, render_template,flash , redirect
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder="template")

pickle_heart = open("classifier_heart.pkl", "rb")
classifier_heart = pickle.load(pickle_heart)

pickle_diabetes = open("classifier_diabetes.pkl", "rb")
classifier_diabetes = pickle.load(pickle_diabetes)

classifier_malaria = load_model("malaria2.h5")
classifier_tuberculosis = load_model("tuberculosis.h5")


# home page
@app.route("/")
def home():
    return render_template("home.html")

# heart failure page
@app.route("/heart")
def heart():
    return render_template("heart_failure.html")

# diabetes page
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

# malaria page
@app.route("/malaria", methods=["GET"])
def malaria():
    return render_template("malaria.html")

# tuberculosis page
@app.route("/tuberculosis")
def tuberculosis():
    return render_template("tuberculosis.html")


# ------------Predicting HEART_FAILURE and DIABETES (machine Learning)---------------------------------------

def prediction(inputs_values, input_size):
    input_values_array = np.array(inputs_values).reshape(1, input_size)
    if input_size == 12:
        predicted_value = classifier_heart.predict(input_values_array)
    elif input_size == 8:
        predicted_value = classifier_diabetes.predict(input_values_array)
    return predicted_value[0]


@app.route('/result', methods=["POST"])
def result():
    if request.method == "POST":
        inputs_values = request.form.to_dict()
        inputs_values = list(inputs_values.values())
        inputs_values = list(map(float, inputs_values))

        if len(inputs_values) == 12:     # heart_failure
            predicted_value = prediction(inputs_values, 12)
        elif  len(inputs_values) == 8:   # diabetes
            predicted_value = prediction(inputs_values, 8)

    if int(predicted_value) == 1:
        prediction_text = "There can be a sign of having a disease, so it's a good idea to see your doctor"
    else:
        prediction_text = "There are no dangerous symptoms of a disease. Stay Safe...Stay Healthy!!!"
    return render_template("result_HeartDiabetes.html", prediction_text=prediction_text)


# ------------------Predicting MALARIA and TUBERCULOSIS (Deep Learning)-----------------------------------

# MALARIA PREDICTION
# function to preprocess the selected image and to predict it's class
def predict_malaria(image_path, classifier_malaria):
    print(image_path)
    selected_image = image.load_img(image_path, target_size = (100, 100, 3))
    selected_image = image.img_to_array(selected_image)
    selected_image = selected_image/255
    selected_image = np.expand_dims(selected_image, axis=0)

    predicted = classifier_malaria.predict(selected_image)
    predicted = np.argmax(predicted, axis=1)
    if predicted == 0:
        predicted = "Cell is Parasitized"
    else:
        predicted = "Cell is Uninfected"

    return predicted


# api for predict button
@app.route("/predict_m", methods=["POST", "GET"])
def predict_m():
    if request.method == "POST":
        # fetching the selected image named "image". This name was given in html <input type="file">
        image_fetch = request.files["image"]

        # saving the selected image in folder named "uploads"
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "uploads", secure_filename(image_fetch.filename))
        image_fetch.save(file_path)

        # Now using the new image path from "uploads" folder to predict the image class
        result = predict_malaria(file_path, classifier_malaria)
        print(result)

        return render_template('result_malaria.html', prediction_text=result)


# TUBERCULOSIS PREDICTION
def predict_tuberculosis(image_path, classifier_tuberculosis):
    print(image_path)
    selected_image = image.load_img(image_path, target_size=(300, 300, 3))
    selected_image = image.img_to_array(selected_image)
    selected_image = selected_image / 255
    selected_image = np.expand_dims(selected_image, axis=0)

    predicted = classifier_tuberculosis.predict(selected_image)
    predicted = np.argmax(predicted, axis=1)
    if predicted == 0:
        predicted = "There is no sign of Tuberculosis. Stay Safe...Stay Healthy!!!"
    else:
        predicted = "There can be a sign of Tuberculosis."

    return predicted


# api for predict button
@app.route("/predict_t", methods=["POST", "GET"])
def predict_t():
    if request.method == "POST":
        # fetching the selected image named "image". This name was given in html <input type="file">
        image_fetch = request.files["image"]

        # saving the selected image in folder named "uploads"
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "uploads", secure_filename(image_fetch.filename))
        image_fetch.save(file_path)

        # Now using the new image path from "uploads" folder to predict the image class
        result = predict_tuberculosis(file_path, classifier_tuberculosis)
        print(result)

        return render_template('result_tuberculosis.html', prediction_text=result)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory("uploads", filename)







if __name__ == "__main__":
    app.run(debug=True)







