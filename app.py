from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import base64
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('braintumor.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_bytes = img.read()
        img_array = np.array(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150))
        img_array = np.expand_dims(img, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        predictions = model.predict(img_array)
        indices = np.argmax(predictions)
        probabilities = np.max(predictions)
        labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        result = {'label': labels[indices], 'probability': float(probabilities)}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)