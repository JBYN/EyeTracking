# source
# https://www.youtube.com/watch?v=PmZ29Vta7Vc
# https://docs.opencv.org/3.4.3/
"""
Created on Tue Jan 15 16:25:07 2019

@author: Jo
"""

import cv2
import os
import pickle

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_label = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # images with the same label are collected in one folder with the label as name
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            # image processing for training
            pil_image = Image.open(path).convert("L")  # conversion to grayscale
            size = (550, 550)
            resized_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(resized_image, "uint8")  # array with BGR values
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]  # region of interest
                x_train.append(roi)
                y_label.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")
