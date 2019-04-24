# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:30:50 2019

@author: Jo
"""
from enum import Enum

import cv2
import dlib
import numpy as np
import constants as cons
from constants import Point, SCREEN_HEIGHT, SCREEN_WIDTH


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

    def getLeftUpperCorner(self) -> cons.Point:
        return cons.Point(self.x, self.y)

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height


class EyePupil:

    def __init__(self, posEye: cons.Point, centerPupil: cons.Point):
        self.posPupil = centerPupil
        self.posEye = posEye
        self.globalPosition = cons.Point(centerPupil.x + posEye.x, centerPupil.y + posEye.y)

    def getPosEyeCenter(self) -> cons.Point:
        return self.posPupil

    def getGlobalPosition(self) -> cons.Point:
        return self.globalPosition


class Eye:
    def __init__(self, img: np.ndarray, positionEye: Rectangle, faceMarks):
        self.pEye = positionEye
        self.img = img
        self.faceMarks = faceMarks
        self.roiEye = img[positionEye.getLeftUpperCorner().y:positionEye.getLeftUpperCorner().y + positionEye.getHeight(),
                      positionEye.getLeftUpperCorner().x:positionEye.getLeftUpperCorner().x + positionEye.getWidth()]
        self.eyePupil = None
        self.main()

    def main(self):
        self.createPupil()

    # def getFaceImage(self) -> np.ndarray:
    #     return self.imgFace

    def getPosEye(self) -> Rectangle:
        return self.pEye

    def getPupil(self) -> EyePupil:
        return self.eyePupil

    def createPupil(self):
        # preprocessing
        blur_Eye = cv2.GaussianBlur(self.roiEye, (cons.BLUR_WEIGHT_SIZE, cons.BLUR_WEIGHT_SIZE), 0, 0)
        _, threshold = cv2.threshold(blur_Eye, cons.PUPIL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get position pupil
        if contours is not None:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            # DEBUG
            # cv2.drawContours(threshold, contours, -1, (0, 255, 0), 3)
            cv2.namedWindow("DEBUG", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("DEBUG", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("DEBUG", threshold)
            for cnt in contours:
                print("PUPIL_DETECTED")
                (x, y, w, h) = cv2.boundingRect(cnt)
                self.eyePupil = EyePupil(self.getPosEye().getLeftUpperCorner(), cons.Point(int(x + w / 2), int(y + h / 2)))


class Face:

    def __init__(self, img: np.ndarray, positionFace: dlib.rectangle):
        self.img = img
        self.pFace = positionFace
        self.roiFace = img[positionFace.top():positionFace.bottom(),
                        positionFace.left():positionFace.right()]  # pixels of the region of interest
        self.faceMarks = cons.FACEMARK_PREDICTOR(img, positionFace)

        self.leftEye = None
        self.posLeftEyeCorner = cons.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)
        self.rightEye = None
        self.posRightEyeCorner = cons.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)

        self.main()

    def main(self):
        # find the positions for the left and right eye
        pL, pR = self.findEyeRegions()
        # create object from class Eye for left and right eye
        self.createEyes(pL, pR)

    def getPosFace(self) -> Rectangle:
        return Rectangle(self.pFace.left(), self.pFace.top(), self.pFace.right() - self.pFace.left(),
                         self.pFace.bottom() - self.pFace.top())

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

    def getPosLeftEyeCorner(self):
        return self.posLeftEyeCorner

    def getPosRightEyeCorner(self):
        return self.posRightEyeCorner

    def setLeftEye(self, eye: Eye):
        self.leftEye = eye

    def setRightEye(self, eye: Eye):
        self.rightEye = eye

    def findEyeRegions(self):
        # Left eye region
        leftEye_x = self.faceMarks.part(42).x
        leftEye_y = self.faceMarks.part(44).y
        leftEye_width = self.faceMarks.part(45).x - leftEye_x
        leftEye_height = self.faceMarks.part(47).y - leftEye_y
        posLeftEye = Rectangle(leftEye_x, leftEye_y, leftEye_width, leftEye_height)

        # Right eye region
        rightEye_x = self.faceMarks.part(36).x
        rightEye_y = self.faceMarks.part(38).y
        rightEye_width = self.faceMarks.part(39).x - rightEye_x
        rightEye_height = self.faceMarks.part(40).y - rightEye_y
        posRightEye = Rectangle(rightEye_x, rightEye_y, rightEye_width, rightEye_height)

        return posLeftEye, posRightEye

    def createEyes(self, posLeftEye: Rectangle, posRightEye: Rectangle):
        self.leftEye = Eye(self.img, posLeftEye, self.faceMarks)
        self.rightEye = Eye(self.img, posRightEye, self.faceMarks)

    @staticmethod
    def findEyeVector(eye: Eye, posEyeCorner: cons.Point) -> cons.Point:
        if eye.getPupil() is not None:
            x = eye.getPupil().getGlobalPosition().x - posEyeCorner.x
            y = eye.getPupil().getGlobalPosition().y - posEyeCorner.y
            return cons.Point(x, y)
        return None


class ScreenAreaBoundary(Enum):
    W1 = SCREEN_WIDTH
    W2 = int(2*SCREEN_WIDTH/3)
    W3 = int(SCREEN_WIDTH/3)
    W4 = 0

    H1 = SCREEN_HEIGHT
    H2 = int(2*SCREEN_HEIGHT/3)
    H3 = int(SCREEN_HEIGHT)
    H4 = 0
    # LT = Rectangle(0, 0, int(SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3))
    # MT = Rectangle(int(SCREEN_WIDTH/3), 0, int(SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3))
    # RT = Rectangle(int(2*SCREEN_WIDTH/3), 0, SCREEN_WIDTH-(2*int(SCREEN_WIDTH/3)), int(SCREEN_HEIGHT/3))
    # LM = Rectangle(0, int(SCREEN_HEIGHT/3), int(SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3))
    # MM = Rectangle(int(SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3), int(SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3))
    # RM = Rectangle(int(2*SCREEN_WIDTH/3), int(SCREEN_HEIGHT/3), SCREEN_WIDTH-(2*int(SCREEN_WIDTH/3)), int(SCREEN_HEIGHT/3))
    # LD = Rectangle(0, int(2*SCREEN_HEIGHT/3), int(SCREEN_WIDTH/3), SCREEN_HEIGHT-(2*int(SCREEN_HEIGHT/3)))
    # MD = Rectangle(int(SCREEN_WIDTH/3), int(2*SCREEN_HEIGHT/3), int(SCREEN_WIDTH/3), SCREEN_HEIGHT-(2*int(SCREEN_HEIGHT/3)))
    # RD = Rectangle(int(2*SCREEN_WIDTH/3), int(2*SCREEN_HEIGHT/3), SCREEN_WIDTH-(2*int(SCREEN_WIDTH/3)), SCREEN_HEIGHT-(2*int(SCREEN_HEIGHT/3)))


# ToDo redesign the Enum
class CentersScreenArea(Enum):
    LT = Point(int(SCREEN_WIDTH/6), int(SCREEN_HEIGHT/6))
    MT = Point(int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/6))
    RT = Point(int(5*SCREEN_WIDTH/6), int(SCREEN_HEIGHT/6))
    LM = Point(int(SCREEN_WIDTH/6), int(SCREEN_HEIGHT/2))
    MM = Point(int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2))
    RM = Point(int(5*SCREEN_WIDTH/6), int(SCREEN_HEIGHT/2))
    LD = Point(int(SCREEN_WIDTH/6), int(5*SCREEN_HEIGHT/6))
    MD = Point(int(SCREEN_WIDTH/2), int(5*SCREEN_HEIGHT / 6))
    RD = Point(int(5*SCREEN_WIDTH/6), int(5*SCREEN_HEIGHT/6))