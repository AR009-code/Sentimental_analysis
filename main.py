import cv2, csv
import numpy as np
import face_recognition
import os
from datetime import datetime as dt
from deepface import DeepFace

path = './ImagesDB' # path for sample images
imageList = []
emotion_stat=[]
classNames = []
date= dt.today()    # date module function
date_str= date.strftime("%d %b, %Y")    # typecasting dateTime type to string format

def findEncodings(images):  # returning the encoded values of the list of images by using face_recognition module
        encodeList = []
        for frame in images:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(frame)[0]
            encodeList.append(encode)
            return encodeList   # returning the list of arrays of final encoded values of the images


def camCapture():

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #This 'XML' file contains a pre-trained model that was created through extensive training and uploaded 
#by Rainer Lienhart on behalf of Intel in 2000. 
#Rainer's model makes use of the Adaptive Boosting Algorithm (AdaBoost) in order to yield better results and accuracy. 

    myList = os.listdir(path)   # extracting the list of images for once from the path

    for cl in myList:
        curframe = cv2.imread(f'{path}/{cl}')   # This library function is from open cv module , where it reads the pixels of all the images from mylist
        imageList.append(curframe)
        classNames.append(os.path.splitext(cl)[0])
    
    encodeListKnown = findEncodings(imageList)  
    print(len(encodeListKnown))

    cap = cv2.VideoCapture(0)   # an opencv module that opens the webcam that captures an array of incoming frames 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    # cap.set(cv2.CAP_PROP_FPS, 90)
    return face_analyse(cap, encodeListKnown, face_cascade)

def face_analyse(cap, encodeListKnown, face_cascade):

    while True: # event loop

        success, frame = cap.read()
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False) # DeepFace.analyze uses the haarcascade_frontalface_default.xml file to map the current face pixels with its predefined values
        '''Haar cascade is an algorithm that can detect objects in images, irrespective of their scale in image and location. 
 This algorithm is not so complex and can run in real-time. We can train a haar-cascade detector to detect various 
 objects like cars, bikes, buildings, fruits, etc. '''
        emotion_stat.append(result)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),2)  # graphics that outline the part of face 
            cv2.rectangle(frame, (x1, y2 - 35), (x2+50, y2), (0, 255, 0),cv2.FILLED)
            cv2.putText(frame, name, (x1 + 5, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1 -30, y2), (x2 + len(date_str) + 40, y2+40), (255, 0, 0),cv2.FILLED)
            cv2.putText(frame, date_str, (x1 - 30, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)
        if key == ord('e'):    # Press 'e' to escape the window
            break
    cv2.destroyAllWindows()

    table= np.array(emotion_stat)

    return table
    
def createCSV(emotion_stat):
    with open('sentiment_result.csv','w',newline='') as f:
        expression=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral','dominant_emotion']
        writer= csv.DictWriter(f, fieldnames=expression)
        writer.writeheader()
        for face in emotion_stat:
            face['emotion'].update({'dominant_emotion':face['dominant_emotion']})
            writer.writerow(face['emotion'])
