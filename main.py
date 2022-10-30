import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime as dt
from deepface import DeepFace
#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = './ImagesDB'
imageList = []
classNames = []
myList = os.listdir(path)

date= dt.today()
date_str= date.strftime("%d %b, %Y")

for cl in myList:
    curframe = cv2.imread(f'{path}/{cl}')
    imageList.append(curframe)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for frame in images:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(frame)[0]
        encodeList.append(encode)
    return encodeList
    
encodeListKnown = findEncodings(imageList)
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FPS, 60)
# print(cap.get(cv2.CAP_PROP_FPS))
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
while True:
 success, frame = cap.read()
 result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.1, 4)

 for (x, y, w, h) in faces:
  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 font = cv2.FONT_HERSHEY_SIMPLEX
 cv2.putText(frame, result['dominant_emotion'], (50, 50),font, 3, (0, 0, 255), 2, cv2.LINE_4)
 #cv2.imshow('Original Video', frame)
 frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
 frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
 facesCurFrame = face_recognition.face_locations(frameS)
 encodesCurFrame =face_recognition.face_encodings(frameS, facesCurFrame)

 for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):

  matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
  faceDis =face_recognition.face_distance(encodeListKnown,encodeFace)
  matchIndex = np.argmin(faceDis)

  if matches[matchIndex]:

   name = classNames[matchIndex].upper()
   y1, x2, y2, x1 = faceLoc
   y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),2)
   cv2.rectangle(frame, (x1, y2 - 35), (x2+50, y2), (0, 255, 0),cv2.FILLED)
   cv2.putText(frame, name, (x1 + 5, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
   cv2.rectangle(frame, (x1 -30, y2), (x2 + len(date_str) + 40, y2+40), (255, 0, 0),cv2.FILLED)
   cv2.putText(frame, date_str, (x1 - 30, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

 cv2.imshow('Webcam', frame)
 key = cv2.waitKey(1)
 if key == ord('e'):
  break