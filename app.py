import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input #, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Some utilites
import numpy as np
from util import base64_to_pil
import h5py
#tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()

global graph
#graph = tf.get_default_graph()

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print("path", THIS_FOLDER)
#MODEL_PATH = "/app/models/my_model.h5"
print(os.listdir())
print(os.listdir('./models/'))
#MODEL_PATH = '/app/models'
MODEL_PATH = '.\models\my_model.h5'

# Load your own trained model
global model
model = load_model('/app/models/my_model1.h5')
#model = tf.saved_model.load('models')
#model = load_model(os.path.join(os.getcwd(), 'models', 'my_model.h5'))
#model = load_model(os.path.join('models', 'my_model.h5'))
#model = model.model()
#model = tf.saved_model.load(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        pred_class = np.argmax(preds, axis=1)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        if pred_class == 0:
            result = 'Not subluxed'
        elif pred_class == 1:
            result = 'Subluxed'

        #result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    #app.run(debug=False, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
