# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:30:50 2019

@author: Jo
"""
import cv2
import dlib
import numpy as np
import constants as cons
from constants import Point


class EyeParameters:

    def __init__(self, face_cascade: cv2.CascadeClassifier, eye_cascade: cv2.CascadeClassifier, view: bool,
                 image: np.ndarray):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.image = image
        self.view = view


class Rectangle:

    def __init__(self, x_left_upperCorner: int, y_left_upperCorner: int, width: int, height: int):
        self.x = x_left_upperCorner
        self.y = y_left_upperCorner
        self.width = width
        self.height = height

    def getLeftUpperCorner(self) -> Point:
        return Point(self.x, self.y)

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height


class EyePupil:

    def __init__(self, imgEye: np.ndarray, centerPupil: Point):
        self.posPupil = centerPupil
        self.img = imgEye
        self.globalPosition = None

    def getPosEyeCenter(self) -> Point:
        return self.posPupil

    def getImg(self):
        return self.img

    def getGlobalPosition(self) -> Point:
        return self.globalPosition

    def setGlobalPosition(self, globalPosition: Point):
        self.globalPosition = globalPosition


class Eye:

    def __init__(self, imgFace: np.ndarray, positionEye: Rectangle):
        self.pEye = positionEye
        self.imgFace = imgFace
        self.roiEye = imgFace[self.pEye.getLeftUpperCorner().y:self.pEye.getLeftUpperCorner().y + self.pEye.getHeight(),
                      self.pEye.getLeftUpperCorner().x:self.pEye.getLeftUpperCorner().x + self.pEye.getWidth()]
        self.eyePupil = self.findPupil()
        self.globalPositionEye = None

    def getFaceImage(self) -> np.ndarray:
        return self.imgFace

    def getPosEye(self) -> Rectangle:
        return self.pEye

    def getPupil(self) -> EyePupil:
        return self.eyePupil

    def getGlobalPositionEye(self) -> Point:
        return self.globalPositionEye

    def setGlobalPositionEye(self, pos: Point):
        self.globalPositionEye = pos

    def findPupil(self) -> EyePupil:
        # preprocessing
        blur_Eye = cv2.GaussianBlur(self.roiEye, (cons.BLUR_WEIGHT_SIZE, cons.BLUR_WEIGHT_SIZE), 0, 0)
        _, threshold = cv2.threshold(blur_Eye, 50, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get position pupil
        if contours is not None:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                return EyePupil(self.roiEye, Point(int(x + w / 2), int(y + h / 2)))
        return None


class EyeCorner:

    def __init__(self, img: np.ndarray, p_face: np.ndarray):
        self.img = img
        self.posFace = p_face
        predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        self.faceMarks = predictor(self.img, p_face)

    def getPosLeftEyeCorner(self) -> Point:
        return Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)

    def getPosRightEyeCorner(self) -> Point:
        return Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)


class Face:

    def __init__(self, img: np.ndarray, positionFace: dlib.rectangle):
        self.img = img
        self.pFace = positionFace
        self.posLeftEye = None
        self.posRightEye = None
        self.roiFace = img[positionFace.top():positionFace.bottom(),
                       positionFace.left():positionFace.right()]  # pixels of the region of interest
        self.leftEye = None
        self.rightEye = None
        self.EyeCorner = self.findEyeCorner()

    def getGlobalImage(self) -> np.ndarray:
        return self.img

    def getPosFace(self) -> Rectangle:
        return Rectangle(self.pFace.left(), self.pFace.top(), self.pFace.right()-self.pFace.left(), self.pFace.bottom()-self.pFace.top())

    def getPosLeftEye(self) -> Rectangle:
        return self.posLeftEye

    def getPosRightEye(self) -> Rectangle:
        return self.posRightEye

    def getLeftEye(self):
        return self.leftEye

    def getROIFace(self) -> np.ndarray:
        return self.roiFace

    def getRightEye(self):
        return self.rightEye

    def getEyeCorner(self) -> EyeCorner:
        return self.EyeCorner

    def setPosLeftEye(self, pos: Rectangle):
        self.posLeftEye = pos

    def setPosRightEye(self, pos: Rectangle):
        self.posRightEye = pos

    def setLeftEye(self, eye: Eye):
        self.leftEye = eye

    def setRightEye(self, eye: Eye):
        self.rightEye = eye

    def setROIFace(self, roi: np.ndarray):
        self.roiFace = roi

    def findEyeCorner(self) -> EyeCorner:
        return EyeCorner(self.img, self.pFace)

    def findPosEyes(self) -> bool:
        # preprocessing
        sigma = cons.SMOOTH_FACTOR * (self.pFace.right() - self.pFace.left())
        roi = cv2.GaussianBlur(self.roiFace, (0, 0), sigma)
        self.setROIFace(roi)

        eyes = cons.EYE_CASCADE.detectMultiScale(roi)

        nr_eyes = 0
        if eyes != ():
            # getting the coordinates of the detected eyes
            for (x, y, w, h) in eyes:
                nr_eyes += 1
                if nr_eyes == 1:
                    self.setPosLeftEye(Rectangle(x, y, w, h))
                elif (nr_eyes == 2) and np.abs(self.getPosLeftEye().getHeight() - y) <= 50:
                    self.setPosRightEye(Rectangle(x, y, w, h))
                    return True
                else:
                    nr_eyes -= 1
        return False

    def findEyes(self) -> bool:
        EYES_DETECTED = self.findPosEyes()
        if EYES_DETECTED:
            eye_1 = self.getPosLeftEye()
            eye_2 = self.getPosRightEye()

            if eye_1.x < eye_2.x and (eye_1.x + eye_1.width) < eye_2.x:
                self.setPosRightEye(eye_1)
                self.setPosLeftEye(eye_2)
            elif eye_2.x < eye_1.x and (eye_2.x + eye_2.width) < eye_1.x:
                self.setPosRightEye(eye_2)
                self.setPosLeftEye(eye_1)
            return True
        return False

    @staticmethod
    def findEyeVector(rightEye: Eye, eyeCorner: EyeCorner) -> cons.Point:
        if rightEye.getPupil() is not None:
            x = rightEye.getPupil().getPosEyeCenter().x - eyeCorner.getPosRightEyeCorner().x
            y = rightEye.getPupil().getPosEyeCenter().y - eyeCorner.getPosRightEyeCorner().y
            return cons.Point(x, y)
        return None