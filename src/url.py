import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import urllib.request
from PIL import Image


class URL:
    
    def __init__(self, url, name):
        self.url = url
        self.name = name

    def convertToImage(self):
        try:
            urllib.request.urlretrieve(self.url , self.name)
            img = Image.open(self.name)
        except Exception as e:
            print(e)
        else:
            face_img = cv2.imread(self.name)
            face_img = cv2.resize(face_img, (600, 600))
            return face_img