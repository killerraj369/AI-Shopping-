import cv2 as cv 
from cvzone.ClassificationModule import Classifier

myData = Classifier('keras_model.h5','labels.txt')

cap = cv.VideoCapture(0)

while True:
    _,img = cap.read()
    
    if img is None:
          print("not getting image")
          continue
      
    predict, index = myData.getPrediction(img,color=(0,0,255))
    print(predict,index)
    
    cv.imshow('frame',img)
    key = cv.waitKey(1)
    
    # Press esc key to exit
    if key == 27:
        break