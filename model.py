# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:30:50 2019

@author: Jo
"""
from enum import IntEnum

import cv2
import dlib
import copy
import numpy as np
import constants as cons
from constants import SCREEN_HEIGHT, SCREEN_WIDTH
from datetime import datetime


class FaceParameters:

    def __init__(self, face_cascade: cv2.CascadeClassifier, view: bool, image: np.ndarray):
        self.face_cascade = face_cascade
        self.image = image
        self.view = view


class Rectangle:

    def __init__(self, x_left_upper_corner: int, y_left_upper_corner: int, width: int, height: int):
        self.x = x_left_upper_corner
        self.y = y_left_upper_corner
        self.width = width
        self.height = height

    def getLeftUpperCorner(self) -> cons.Point:
        return cons.Point(self.x, self.y)

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height


class EyePupil:

    def __init__(self, pos_eye: cons.Point, center_pupil: cons.Point):
        self.posPupil = center_pupil
        self.posEye = pos_eye
        self.globalPosition = cons.Point(center_pupil.x + pos_eye.x, center_pupil.y + pos_eye.y)

    def get_pos_eye_center(self) -> cons.Point:
        return self.posPupil

    def get_global_position(self) -> cons.Point:
        return self.globalPosition


class Eye:
    def __init__(self, img: np.ndarray, position_eye: Rectangle, face_marks):
        self.pEye = position_eye
        self.img = img
        self.faceMarks = face_marks
        self.roiEye = img[
                      position_eye.getLeftUpperCorner().y:position_eye.getLeftUpperCorner().y + position_eye.getHeight(),
                      position_eye.getLeftUpperCorner().x:position_eye.getLeftUpperCorner().x + position_eye.getWidth()]
        self.eyePupil = None
        self.main()

    def main(self):
        self.create_pupil()

    def get_roi(self) -> np.ndarray:
        return self.roiEye

    def get_pos_eye(self) -> Rectangle:
        return self.pEye

    def get_pupil(self) -> EyePupil:
        return self.eyePupil

    def create_pupil(self):
        # pre-processing
        # blur_Eye = cv2.GaussianBlur(self.roiEye, (cons.BLUR_WEIGHT_SIZE, cons.BLUR_WEIGHT_SIZE), 0, 0)
        _, threshold = cv2.threshold(self.roiEye, cons.PUPIL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        # collect data to get proper threshold value
        if cons.COLLECT_DATA:
            # DEBUG
            # cv2.namedWindow("Eye", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("Eye", self.roiEye)

            # cv2.namedWindow("Eye_Threshold", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Eye_Threshold", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("Eye_Threshold", threshold)

            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print("C:" + str(contours))
            # TODO collect data
            cons.threshold.append(threshold)
            # TODO size eyeregion -> adjust calibration
            size = self.roiEye.size

        # get position pupil
        if contours:
            print("CONT")
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                print("PUPIL_DETECTED")
                (x, y, w, h) = cv2.boundingRect(cnt)
                self.eyePupil = EyePupil(self.get_pos_eye().getLeftUpperCorner(), cons.Point(int(x + w / 2), int(y + h / 2)))

                # collect data to set threshold for light intensity
                if cons.COLLECT_DATA:
                    cons.area.append(cv2.contourArea(cnt) / size)
                    copy_eye = copy.deepcopy(self.roiEye)
                    a = cv2.circle(copy_eye,
                                   (self.eyePupil.get_pos_eye_center().x, self.eyePupil.get_pos_eye_center().y),
                                   cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)
                    # cv2.namedWindow("IRIS_Threshold", cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty("IRIS_Threshold", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow("IRIS_Threshold", self.roiEye)
                    cons.eyes.append(a)
                break
        else:
            if cons.COLLECT_DATA:
                cons.area.append(0.0)
                cons.eyes.append(self.roiEye)


class Face:

    def __init__(self, img: np.ndarray, position_face: dlib.rectangle):
        self.img = img
        self.pFace = position_face
        self.faceMarks = cons.FACEMARK_PREDICTOR(img, position_face)

        self.leftEye = None
        self.posLeftEyeCorner = cons.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)
        self.rightEye: Eye = None
        self.posRightEyeCorner = cons.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)

        self.main()

    def main(self):
        # find the positions for the left and right eye
        p_l, p_r = self.findEyeRegions()
        # create object from class Eye for left and right eye
        self.createEyes(p_l, p_r)

    def update(self, img: np.ndarray) -> bool:
        self.img = img
        self.faceMarks = cons.FACEMARK_PREDICTOR(img, self.pFace)
        # TODO When the face isn't in the area, eyes are indicated on non-faces
        if self.faceMarks is None:
            return False
        else:
            self.posLeftEyeCorner = cons.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)
            self.posRightEyeCorner = cons.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)
        self.main()

    def getPosFace(self) -> Rectangle:
        return Rectangle(self.pFace.left(), self.pFace.top(), self.pFace.right() - self.pFace.left(),
                         self.pFace.bottom() - self.pFace.top())

    def getPosLeftEye(self) -> Rectangle:
        return self.posLeftEye

    def getPosRightEye(self) -> Rectangle:
        return self.posRightEye

    def getLeftEye(self):
        return self.leftEye

    def getRightEye(self) -> Eye:
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
        # self.leftEye = Eye(self.img, posLeftEye, self.faceMarks)
        self.rightEye = Eye(self.img, posRightEye, self.faceMarks)

    @staticmethod
    def findEyeVector(eye: Eye, posEyeCorner: cons.Point) -> cons.Point:
        if eye.get_pupil() is not None:
            x = eye.get_pupil().get_global_position().x - posEyeCorner.x
            y = eye.get_pupil().get_global_position().y - posEyeCorner.y
            print("Vector: " + str(x) + ", " + str(y))
            return cons.Point(x, y)
        return None


class ScreenAreaBoundary(IntEnum):
    W1 = SCREEN_WIDTH
    W2 = 2*SCREEN_WIDTH/3
    W3 = SCREEN_WIDTH/3
    W4 = 0

    H1 = SCREEN_HEIGHT
    H2 = 2*SCREEN_HEIGHT/3
    H3 = SCREEN_HEIGHT
    H4 = 0


class CenterXScreenArea(IntEnum):
    L = int(SCREEN_WIDTH/6)
    M = int(SCREEN_WIDTH/2)
    R = int(5*SCREEN_WIDTH/6)


class CenterYScreenArea(IntEnum):
    T = int(SCREEN_HEIGHT/6)
    M = int(SCREEN_HEIGHT/2)
    D = int(5*SCREEN_HEIGHT/6)
