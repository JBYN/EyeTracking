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


