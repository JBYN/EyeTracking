# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")  # importing the training data

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cam = cv2.VideoCapture(0)  # 0 used as parameter to use the webcam of the computer/laptop

while True:
    b, img = cam.read()
    # to make it possible to detect faces the capture has to be in grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0 | cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT)
    if faces == ():
        cv2.putText(img, "Not able to detect a face", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # pixels of the region of interest
            roi_color = img[y:y + h, x:x + w]

            # recognizing
            # confidence is the distance in the training model. The closer the object is the more sure the program is.
            id_, conf = recognizer.predict(roi_gray)
            # print("id: " + labels[id_])
            # print("conf: " + str(conf))
            # if not conf <= 50:
            #    name = "unknow"
            # else:
            name = labels[id_] + " ---->  " + str(conf)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0)
            stroke = 2
            cv2.putText(img, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)

            starting_coord = (x, y)
            ending_coord = (x + w, y + h)
            color = (0, 255, 0)  # BGR
            stroke = 2
            cv2.rectangle(img, starting_coord, ending_coord, color, stroke)

    if b:
        cv2.imshow("Window", img)
    else:
        print("The camera is not working")
        break
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
