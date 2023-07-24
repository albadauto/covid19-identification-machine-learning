from flask import Flask, jsonify, request
import os
import tensorflow as tf
import cv2
import numpy as np
app = Flask(__name__)
model = tf.keras.models.load_model("H:/Python/MachineLearning/1/PROJETOS/PROJETO_COVID_19/model.h5")

dict = {
    0: "Covid",
    1: "Opacidade pulmonar",
    2: "Normal",
    3: "Pneumonia viral"
}

@app.route("/helloworld", methods=['GET'])
def helloworld():
    return "teste"


@app.route("/predict", methods=['POST'])
def predict():
    image = request.files['image']
    image.save(os.path.join("image/", image.filename))
    image_array = cv2.imread("image/" + image.filename)
    image_array = cv2.resize(image_array, (32, 32))
    image_array = np.array(image_array)
    image_array = image_array / 255.0
    image_array = np.sum(image_array / 3, axis=2, keepdims=True)
    image_array = np.expand_dims(image_array, axis=(0, 3))
    prediction = dict[np.argmax(model.predict(image_array), axis=1)[0]]
    return jsonify({"data": prediction})


if __name__ == '__main__':
    app.run("localhost", debug=True)