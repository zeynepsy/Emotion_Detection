from logging import exception
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import urllib.request
from PIL import Image



class showResult:
    
    def __init__(self, img):
        self.img = img

    def show(self):
        try:
            while True:
                    result = cv2.imshow('Emotions', self.img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        except Exception as e:
            print(e)
           
    cv2.destroyAllWindows()