#!/usr/bin/python

# test file
# TODO:
# 	Figure out four point transform
#	Figure out testing data warping
# 	Use webcam as input
# 	Figure out how to use contours
# 		Currently detects inner rect -> detect outermost rectangle
# 	Try using video stream from android phone


#from utils import *
from matplotlib import pyplot as plt
from prediction import prediction
import subprocess
from gtts import gTTS
import cv2
import requests


"""
def rates(value):
  r = requests.get('http://data.fixer.io/api/latest?access_key=3d4375b4c1d93a7f9cacc0e720c1c24c&format=1')

  rupee = r.json()
  eur1 = rupee['rates']['INR']
  usd1 = rupee['rates']['USD']
  rupe= eur1/usd1
  return(value/rupe)
  
"""


obj = prediction()
test_img = cv2.imread('files/test_100_2.jpg')
#plt.imshow(test_img)
val = obj.output(test_img)
print(val)


  