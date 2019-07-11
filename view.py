# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:03:37 2019

@author: Jo
"""

import cv2
import numpy as np
import model as c
import constants as cons


def showImage(img: np.ndarray, face: c.Face, name: str):
    drawEyePupils(img, face)
    drawEyeCorners(img, face)
    drawEyeRegion(img, face)
    show(img, name)


def drawEyeCorners(img: np.ndarray, face: c.Face):
    cv2.circle(img, (face.getPosLeftEyeCorner().x, face.getPosLeftEyeCorner().y),
               cons.RADIUS_EYECORNER_IND, cons.COLOR_EYECORNER_IND)
    cv2.circle(img, (face.getPosRightEyeCorner().x,
                     face.getPosRightEyeCorner().y), cons.RADIUS_EYECORNER_IND, cons.COLOR_EYECORNER_IND)


def drawEyePupils(img: np.ndarray, face: c.Face):
    #c_l = face.getLeftEye().getPupil().getGlobalPosition()
    #cv2.circle(img, (c_l.x, c_l.y), cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)

    c_r = face.getRightEye().get_pupil().get_global_position()
    cv2.circle(img, (c_r.x, c_r.y), cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)


def drawEyeRegion(img: np.ndarray, face: c.Face):
    # left eye region
    x_l = face.getLeftEye().get_pos_eye().getLeftUpperCorner().x
    y_l = face.getLeftEye().get_pos_eye().getLeftUpperCorner().y
    cv2.rectangle(img, (x_l, y_l),
                  (x_l + face.getLeftEye().get_pos_eye().width, y_l + face.getLeftEye().get_pos_eye().height),
                  cons.COLOR_EYE_IND)

    # right eye region
    x_r = face.getRightEye().get_pos_eye().getLeftUpperCorner().x
    y_r = face.getRightEye().get_pos_eye().getLeftUpperCorner().y
    cv2.rectangle(img, (x_r, y_r),
                  (x_r + face.getRightEye().get_pos_eye().width, y_r + face.getRightEye().get_pos_eye().height),
                  cons.COLOR_EYE_IND)


def show(img: np.ndarray, name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, img)


def showPos(pos: cons.Point):
    img = cons.createBlankScreen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    cv2.circle(img, (pos.x, pos.y), cons.RADIUS_LOOK_POINT, cons.COLOR_LOOK_POINT,
               cons.THICKNESS_LOOK_POINT)
    show(img, "Position")

