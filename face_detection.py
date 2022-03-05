import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import urllib.request
from PIL import Image


class faceDetection:

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    def __init__(self, face_img):
        self.face_img = face_img

    def createRects(self, my_img, predicted_emotion):
        try:
            face_rects= self.faceCascade.detectMultiScale(self.face_img)
        
            for (x,y,w,h) in face_rects:
                
                    cv2.rectangle(my_img,(x,y),(x+w,y+h),(0,255,0),5)
                    cv2.putText(my_img, predicted_emotion, (int(x),int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                

        except Exception as e:
            print(e)

        else:
            return my_img
        