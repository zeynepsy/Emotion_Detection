

from URL import URL
from emotion_detection import emotionDetection
from face_detection import faceDetection
from show_result import showResult
#import cv2

obj1 = URL('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRf3ARBNKN7dzgV72l2G1dnNVxcJGP39M22Cw&usqp=CAU', 'img.jpg')
my_img = obj1.convertToImage()

obj3 = emotionDetection(my_img)
predictedEmotion = obj3.predictEmotion()

obj2 = faceDetection(my_img)
img = obj2.createRects(my_img, predictedEmotion)

obj4 = showResult(img)
obj4.show()

#cv2.destroyAllWindows()