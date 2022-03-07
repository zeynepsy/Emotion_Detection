import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import urllib.request
from PIL import Image


class emotionDetection:

    def __init__(self, face_img):
        self.face_img = face_img

    def predictEmotion(self):
        try:
            predictions = DeepFace.analyze(self.face_img, actions = ['emotion'])
        except Exception as e:
            print(e)
        else:
            predicted_emotion =predictions['dominant_emotion']
            return predicted_emotion