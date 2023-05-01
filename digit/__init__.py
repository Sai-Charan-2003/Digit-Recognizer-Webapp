import tensorflow as tf
from keras.models import load_model
import os
import cv2
import numpy as np
from flask import Flask, render_template, request


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def inputImg():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            img = request.files['imginp'].read()
            img_bytes = np.fromstring(img, np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            _, thresh = cv2.threshold(image, 80, 225, cv2.THRESH_BINARY_INV)
            APP_ROOT = os.path.dirname(os.path.abspath(__file__))
            MODEL = os.path.join(APP_ROOT, 'model.h5')
            model = load_model(MODEL)
            rimg = np.array(thresh) / 255.0
            t = tf.convert_to_tensor(rimg[np.newaxis, :, :], dtype='float32')
            p = model.predict(t)
            return render_template('predict_page.html',
                                   value='The predicted number is {}'.format(np.argmax(p, axis=1)[0]))

    return app
