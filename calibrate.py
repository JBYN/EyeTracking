# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:39:50 2019

@author: Jo
"""

import cv2
import numpy as np
import statistics as stat
import constants as cons
import model as m
import view


class CalibrateScreen:

    def __init__(self, screen: np.ndarray, posPoint: cons.Point):
        self.point = posPoint
        self.screen = screen
        self.main()

    def getPositionPoint(self) -> cons.Point:
        return self.point

    def getScreen(self) -> np.ndarray:
        return self.screen

    def main(self):
        cv2.circle(self.screen, (self.point.x, self.point.y), cons.RADIUS_CALIBRATE_POINT, cons.COLOR_CALIBRATE_POINT,
                   cons.THICKNESS_CALIBRATE_POINT)


class CalibrateLightIntensity:
    # TODO works but is finished before the screen with spot to look is showed. Time delay, delays showing screen.
    def __init__(self, face, calibrateScreen: CalibrateScreen):
        self.face = face
        self.calibrateScreen = calibrateScreen
        self.threshold = 0
        self.main()

    def getThreshold(self):
        return self.threshold

    def main(self):
        view.show(self.calibrateScreen.getScreen(), cons.NAME_CALIBRATE_WINDOW)
        eye = self.face.getRightEye().getROI()
        # pre-process
        #blur_Eye = cv2.GaussianBlur(eye, (cons.BLUR_WEIGHT_SIZE, cons.BLUR_WEIGHT_SIZE), 0, 0)
        self.findThreshold(self.threshold, cons.AREA_THRESHOLD, eye)

    def findThreshold(self, lightThreshold: int, areaThreshold: int, eye: m.Eye):
        area = 0
        _, threshold = cv2.threshold(eye, lightThreshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None:
            self.threshold += 2
            self.findThreshold(self.threshold, areaThreshold, eye)
        else:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                print("PUPIL_DETECTED!")
                area = cv2.contourArea(cnt)
                break
            if area >= areaThreshold:
                return None
            else:
                self.threshold += 2
                self.findThreshold(self.threshold, areaThreshold, eye)

        return None


class Calibrate:

    def __init__(self, calibrateScreen: CalibrateScreen, face):
        self.calibrateScreen = calibrateScreen
        self.face = face
        self.vectorX = list()
        self.vectorY = list()
        self.numberOfCalibrateData = 0
        self.main()

    def main(self):
        view.show(self.calibrateScreen.getScreen(), cons.NAME_CALIBRATE_WINDOW)
        self.findVector()

    def getCalibrateScreen(self) -> CalibrateScreen:
        return self.calibrateScreen

    def getVectorX(self) -> int:
        return int(stat.mean(self.vectorX))

    def getVectorY(self) -> int:
        return int(stat.mean(self.vectorY))

    def setFace(self, face: m.Face):
        self.face = face

    def updateVectors(self, vector: cons.Point):
        self.vectorX.append(vector.x)
        self.vectorY.append(vector.y)

    def findVector(self):
        vector = self.findEyeVector(self.face.getRightEye(), self.face.posRightEyeCorner)
        if vector is not None:
            self.updateVectors(vector)
            self.numberOfCalibrateData += 1
            print("NUMBER: " + str(self.numberOfCalibrateData))

    def updateCalibrate(self, face: m.Face):
        self.face = face
        self.findVector()

    def findEyeVector(self, eye: m.Eye, posEyeCorner: cons.Point) -> cons.Point:
        return self.face.findEyeVector(eye, posEyeCorner)


