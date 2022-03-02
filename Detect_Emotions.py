from unicodedata import name
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
        urllib.request.urlretrieve(self.url , self.name)
        img = Image.open(self.name)
        face_img = cv2.imread(self.name)
        face_img = cv2.resize(face_img, (600, 600))

        return face_img

class faceDetection:

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    def __init__(self, face_img):
        self.face_img = face_img

    def createRects(self, my_img, predicted_emotion):
        face_rects= self.faceCascade.detectMultiScale(self.face_img)
        
        for (x,y,w,h) in face_rects:
            cv2.rectangle(my_img,(x,y),(x+w,y+h),(0,255,0),5)

            cv2.putText(my_img, predicted_emotion, (int(x),int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        return my_img

class emotionDetection:

    def __init__(self, face_img):
        self.face_img = face_img

    def predictEmotion(self):

        predictions = DeepFace.analyze(self.face_img, actions = ['emotion'])
        predicted_emotion =predictions['dominant_emotion']
        
        return predicted_emotion

class showResult:
    
    def __init__(self, img):
        self.img = img

    def show(self):

        while True:
            result = cv2.imshow('Emotions', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
    cv2.destroyAllWindows()



obj1 = URL('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRf3ARBNKN7dzgV72l2G1dnNVxcJGP39M22Cw&usqp=CAU', 'img.jpg')
my_img = obj1.convertToImage()

obj3 = emotionDetection(my_img)
predicted_emotion = obj3.predictEmotion()

obj2 = faceDetection(my_img)
img = obj2.createRects(my_img, predicted_emotion)

obj4 = showResult(img)
obj4.show()




