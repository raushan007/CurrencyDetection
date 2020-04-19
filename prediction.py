
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS
import cv2
import requests

class prediction:  
  
  # read image as is
  def read_img(self,file_name):
    img = cv2.imread(file_name)
    return img


  # resize image with fixed aspect ratio
  def resize_img(self,image, scale):
    res = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    return res
 


  def output(self,test_img):
    max_val = 8
    max_pt = -1
    max_kp = 0
    orb = cv2.ORB_create()
    # orb is an alternative to SIFT
    # resizing must be dynamic
    #original = self.resize_img(test_img, 0.4)
    #display('original', original)

    # keypoints and descriptors
    # (kp1, des1) = orb.detectAndCompute(test_img, None)
    (kp1, des1) = orb.detectAndCompute(test_img, None)

    training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

    for i in range(0, len(training_set)):
      # train image
      train_img = cv2.imread(training_set[i])

      (kp2, des2) = orb.detectAndCompute(train_img, None)

      # brute force matcher
      bf = cv2.BFMatcher()
      all_matches = bf.knnMatch(des1, des2, k=2)

      good = []
      # give an arbitrary number -> 0.789
      # if good -> append to list of good matches
      for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
          good.append([m])

      if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2

      print(i, ' ', training_set[i], ' ', len(good))

    if max_val >= 15:
      print(training_set[max_pt])
      print('Real Image and matched feature is ', max_val)

      train_img = cv2.imread(training_set[max_pt])
      img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
      
      note = str(training_set[max_pt])[-8:]
      split_string = note.split(".")
      fin = split_string[0].split("/")
      l = len(fin)
      print('\nDetected denomination: Rs. ', fin[l-1])
      value = int(fin[l-1])
      #converting the currency value to latest currency value
      
     
      r = requests.get('http://data.fixer.io/api/latest?access_key=3d4375b4c1d93a7f9cacc0e720c1c24c&format=1')
    
      rupee = r.json()
      eur1 = rupee['rates']['INR']
      usd1 = rupee['rates']['USD']
      rupe= round((eur1/usd1),2)
      a = round((value/rupe),2)
      tex = "The currency is real and its value is "+str(value)+". Latest price of dollar is "+str(rupe)+". So the value of currency in dollar will be $" + str(a)              
      #audio_file = 'audio/' + note + '.mp3'

      # audio_file = "value.mp3
      # tts = gTTS(text=speech_out, lang="en")
      # tts.save(audio_file)
      #return_code = subprocess.call(["afplay", audio_file])

      (plt.imshow(img3), plt.show())
      #return int(fin[l-1])
  
      return tex
    else:
      return "fake image"
  
    
    
    