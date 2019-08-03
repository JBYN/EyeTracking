# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:03:37 2019

@author: Jo
"""

import cv2
import numpy as np
import model as c
import constants as cons


def show_image(img: np.ndarray, face: c.Face, name: str):
    draw_eye_pupils(img, face)
    draw_eye_corners(img, face)
    draw_eye_region(img, face)
    show(img, name)


def draw_eye_corners(img: np.ndarray, face: c.Face):
    cv2.circle(img, (face.get_pos_left_eye_corner().x, face.get_pos_left_eye_corner().y),
               cons.RADIUS_EYECORNER_IND, cons.COLOR_EYECORNER_IND)
    cv2.circle(img, (face.get_pos_right_eye_corner().x,
                     face.get_pos_right_eye_corner().y), cons.RADIUS_EYECORNER_IND, cons.COLOR_EYECORNER_IND)


def draw_eye_pupils(img: np.ndarray, face: c.Face):
    c_l = face.get_left_eye().getPupil().getGlobalPosition()
    cv2.circle(img, (c_l.x, c_l.y), cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)

    c_r = face.get_right_eye().get_pupil().get_global_position()
    cv2.circle(img, (c_r.x, c_r.y), cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)


def draw_eye_region(img: np.ndarray, face: c.Face):
    # left eye region
    x_l = face.get_left_eye().get_pos_eye().get_upper_left_corner().x
    y_l = face.get_left_eye().get_pos_eye().get_upper_left_corner().y
    cv2.rectangle(img, (x_l, y_l),
                  (x_l + face.get_left_eye().get_pos_eye().width, y_l + face.get_left_eye().get_pos_eye().height),
                  cons.COLOR_EYE_IND)

    # right eye region
    x_r = face.get_right_eye().get_pos_eye().get_upper_left_corner().x
    y_r = face.get_right_eye().get_pos_eye().get_upper_left_corner().y
    cv2.rectangle(img, (x_r, y_r),
                  (x_r + face.get_right_eye().get_pos_eye().width, y_r + face.get_right_eye().get_pos_eye().height),
                  cons.COLOR_EYE_IND)


def show(img: np.ndarray, name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, img)


def show_pos(pos: cons.Point):
    img = cons.create_blank_screen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    cv2.circle(img, (pos.x, pos.y), cons.RADIUS_LOOK_POINT, cons.COLOR_LOOK_POINT,
               cons.THICKNESS_LOOK_POINT)
    show(img, "Position")
