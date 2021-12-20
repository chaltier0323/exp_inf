#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:09:55 2021

@author: cloudy
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import flask
from flask import jsonify, request
import numpy as np
import time 

app = flask.Flask(__name__)


def model(modelType):
    if modelType=="0":
        model = tf.keras.models.load_model("./mnist")
    else:
        model = tf.keras.models.load_model("./mnistdeep")
    return model

@app.route('/predict', methods=['POST'])
def test():
    data = request.json
    modelType = data["modelType"]
    arr = np.array(data['arr'])
    start = time.time()
    m1 = model(modelType)
    print(time.time() - start)
    b = m1.predict(arr)
    c = np.argmax(b)
    return jsonify(str(c))

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    app.run(debug=True, host='0.0.0.0')

