

from URL import URL
from emotion_detection import emotionDetection
from face_detection import faceDetection
from show_result import showResult
import sys

class Main:

    obj1 = URL(str(sys.argv[1]), 'img.jpg')
    my_img = obj1.convertToImage()

    obj3 = emotionDetection(my_img)
    predictedEmotion = obj3.predictEmotion()

    obj2 = faceDetection(my_img)
    img = obj2.createRects(my_img, predictedEmotion)

    obj4 = showResult(img)
    obj4.show()

