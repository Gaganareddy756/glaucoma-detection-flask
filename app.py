from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import os


app = Flask(__name__)

# ===============================
# LOAD TRAINED MODEL
# ===============================
json_file = open('ImageClassifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('ImageClassifier.weights.h5')

print("Model loaded successfully")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_label(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(150, 150))
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Glaucoma Detected"
    else:
        return "No Glaucoma"

# ===============================
# ROUTES
# ===============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No selected file")

    file_path = "uploaded_image.jpg"
    file.save(file_path)

    result = predict_label(file_path)

    return render_template('index.html', result=result)

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
