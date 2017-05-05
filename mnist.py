from flask import Flask, request, render_template

import sys

import numpy as np
from scipy.misc import imread
from keras.models import load_model

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def index():

    result = None

    if request.method == 'POST':

        # get the uploaded image file
        f = request.files['numberImageFile']

        # convert it to 28x28 matrix
        pic = imread(f)

        # reshape it for keras (entries, rows, cols, channels)
        pic = pic.reshape((1, 28, 28, 1))

        # predict. will get a numpy array with a single prediction
        pred = model.predict(pic)[0]
        # convert single one-hot-encoded prediction to index of label
        # in our case, index == label
        pred = np.argmax(pred)

        # get probability of each prediction
        proba = model.predict_proba(pic)[0]

        result = {
            'pred': pred,
            'proba': proba
        }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    model = load_model('my_model.h5')
    app.run()
