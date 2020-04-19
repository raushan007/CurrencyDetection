# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:37:08 2019

@author: ASUS
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from prediction import prediction

from matplotlib import pyplot as plt
from prediction import prediction
import subprocess
from gtts import gTTS
import cv2
import requests
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        # Save the file to ./uploads
        #basepath = "F:\Temp\Deployment"
		#basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        obj = prediction()
        #test_img = cv2.imread('files/test_100_2.jpg')
        test_img = cv2.imread(file_path)
        #plt.imshow(test_img)
        preds = obj.output(test_img)
        #print(val)
        return preds
    return None

if __name__ == '__main__':
    app.run()

