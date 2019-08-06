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


class FaceParameters:

    def __init__(self, face_cascade: cv2.CascadeClassifier, view: bool, image: np.ndarray):
        self.face_cascade = face_cascade
        self.image = image
        self.view = view


class Rectangle:

    def __init__(self, x_upper_left_corner: int, y_upper_left_corner: int, width: int, height: int):
        self.x = x_upper_left_corner
        self.y = y_upper_left_corner
        self.width = width
        self.height = height

    def get_upper_left_corner(self) -> cons.Point:
        return cons.Point(self.x, self.y)

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height


class EyePupil:

    def __init__(self, pos_eye: cons.Point, region_pupil: Rectangle):
        self.posPupil = region_pupil
        self.posEye = pos_eye
        self.globalPosition = cons.Point(self.get_center().x + self.posEye.x, self.get_center().y + self.posEye.y)
        self.globalROI = Rectangle(region_pupil.x + pos_eye.x, region_pupil.y + pos_eye.y, region_pupil.get_width(),
                                   region_pupil.get_height())

    def get_center(self) -> cons.Point:
        x = (self.posPupil.x + self.posPupil.width)/2
        y = (self.posPupil.y + self.posPupil.height)/2
        return cons.Point(int(x), int(y))

    def get_roi(self) -> Rectangle:
        return self.posPupil

    def get_global_position_center(self) -> cons.Point:
        return self.globalPosition

    def get_global_roi(self) -> Rectangle:
        return self.globalROI


class Eye:
    def __init__(self, img: np.ndarray, position_eye: Rectangle):
        self.pEye = position_eye
        self.img = img
        self.roiEye = img[position_eye.get_upper_left_corner().y:position_eye.get_upper_left_corner().y +
                          position_eye.get_height(),
                          position_eye.get_upper_left_corner().x:position_eye.get_upper_left_corner().x +
                          position_eye.get_width()]
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
        _, threshold = cv2.threshold(self.roiEye, cons.LIGHT_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # collect data to get proper threshold value
        if cons.COLLECT_DATA:
            # DEBUG
            # cv2.namedWindow("Eye", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("Eye", self.roiEye)

            # cv2.namedWindow("Eye_Threshold", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Eye_Threshold", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("Eye_Threshold", threshold)

            cons.threshold.append(threshold)
            size = self.roiEye.size

        # get position pupil
        if contours:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                print("PUPIL_DETECTED")
                (x, y, w, h) = cv2.boundingRect(cnt)
                self.eyePupil = EyePupil(self.get_pos_eye().get_upper_left_corner(), Rectangle(x, y, w, h))
                print("height: " + str(h) + "; width: " + str(w))
                # collect data to set threshold for light intensity
                if cons.COLLECT_DATA:
                    cons.area_pupil.append(cv2.contourArea(cnt))
                    cons.size_eye.append(size)
                    copy_eye = copy.deepcopy(self.roiEye)
                    a = cv2.circle(copy_eye,
                                   (self.eyePupil.get_center().x, self.eyePupil.get_center().y),
                                   cons.RADIUS_PUPIL_IND, cons.COLOR_PUPIL_IND)
                    # cv2.namedWindow("IRIS_Threshold", cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty("IRIS_Threshold", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow("IRIS_Threshold", self.roiEye)
                    cons.eyes.append(a)
                    cons.threshold_value.append(cons.LIGHT_INTENSITY_THRESHOLD)
                break
        else:
            if cons.COLLECT_DATA:
                cons.area_pupil.append(0.0)
                cons.eyes.append(self.roiEye)
                cons.threshold_value.append(cons.LIGHT_INTENSITY_THRESHOLD)
                cons.size_eye.append(size)


class Face:

    def __init__(self, img: np.ndarray, position_face: dlib.rectangle):
        self.img = img
        self.new_img = img
        self.pFace = position_face
        self.faceMarks = cons.FACEMARK_PREDICTOR(img, position_face)

        self.leftEye: Eye = None
        self.posOuterLeftEyeCorner = cons.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)
        self.posInnerLeftEyeCorner = cons.Point(self.faceMarks.part(42).x, self.faceMarks.part(42).y)
        self.rightEye: Eye = None
        self.posOuterRightEyeCorner = cons.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)
        self.posInnerRightEyeCorner = cons.Point(self.faceMarks.part(39).x, self.faceMarks.part(39).y)

        self.main()

    def main(self):
        # find the positions for the left and right eye
        p_l, p_r = self.find_eye_regions()

        # check whether the area of the eye regions is realistic
        if p_l.get_width() * p_l.get_height() > cons.SIZE_EYE and p_r.get_width() * p_r.get_height() > cons.SIZE_EYE:
            # create object from class Eye for left and right eye
            self.create_eyes(p_l, p_r)

            if cons.REFERENCE:
                f = copy.deepcopy(self.new_img)
                le = self.get_pos_face().x
                t = self.get_pos_face().y
                r = le + self.get_pos_face().width
                b = t + self.get_pos_face().height
                cv2.rectangle(f, (le, t), (r, b), (0, 0, 0))
                crop = f[t:b, le:r]
                led_detected = detect_led(crop)
                if led_detected and self.leftEye and self.rightEye:
                    if self.leftEye.get_pupil() and self.rightEye.get_pupil():
                        cons.pupil_position.append(self.leftEye.get_pupil().get_global_position_center().to_string)
                        cons.pupil_position.append(self.rightEye.get_pupil().get_global_position_center().to_string)
                        cons.led_position.append(led_detected.to_string)
                        cons.led_position.append(led_detected.to_string)
                        cons.corner_position.append(self.posOuterLeftEyeCorner.to_string)
                        cons.corner_position.append(self.posOuterRightEyeCorner.to_string)

    def update(self, img: np.ndarray) -> bool:
        movement_detected = self.detect_movement(img)
        if movement_detected:
            return False
        else:
            self.faceMarks = cons.FACEMARK_PREDICTOR(img, self.pFace)
            self.posOuterLeftEyeCorner = cons.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)
            self.posOuterRightEyeCorner = cons.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)
            self.posInnerLeftEyeCorner = cons.Point(self.faceMarks.part(42).x, self.faceMarks.part(42).y)
            self.posInnerRightEyeCorner = cons.Point(self.faceMarks.part(39).x, self.faceMarks.part(39).y)

            self.main()
            return True

    def detect_movement(self, img: np.ndarray) -> bool:
        self.img = copy.deepcopy(self.new_img)
        self.new_img = img
        crop_img = self.img[self.pFace.top():self.pFace.bottom(), self.pFace.left():self.pFace.right()]
        crop_new_img = self.new_img[self.pFace.top():self.pFace.bottom(), self.pFace.left():self.pFace.right()]
        detected_movement = cv2.absdiff(crop_img, crop_new_img)
        _, threshold_movement = cv2.threshold(detected_movement, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold_movement, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    return True
        return False

    def get_pos_face(self) -> Rectangle:
        return Rectangle(self.pFace.left(), self.pFace.top(), self.pFace.right() - self.pFace.left(),
                         self.pFace.bottom() - self.pFace.top())

    def get_left_eye(self) -> Eye:
        return self.leftEye

    def get_right_eye(self) -> Eye:
        return self.rightEye

    def get_pos_outer_left_eye_corner(self) -> cons.Point:
        return self.posOuterLeftEyeCorner

    def get_pos_outer_right_eye_corner(self) -> cons.Point:
        return self.posOuterRightEyeCorner

    def get_pos_inner_left_eye_corner(self) -> cons.Point:
        return self.posInnerLeftEyeCorner

    def get_pos_inner_right_eye_corner(self) -> cons.Point:
        return self.posInnerRightEyeCorner

    def set_left_eye(self, eye: Eye):
        self.leftEye = eye

    def set_right_eye(self, eye: Eye):
        self.rightEye = eye

    def find_eye_regions(self) -> (Rectangle, Rectangle):
        # Left eye region
        left_eye_x = self.faceMarks.part(42).x
        left_eye_y = self.faceMarks.part(44).y
        left_eye_width = self.faceMarks.part(45).x - left_eye_x
        left_eye_height = self.faceMarks.part(47).y - left_eye_y
        pos_left_eye = Rectangle(left_eye_x, left_eye_y, left_eye_width, left_eye_height)

        # Right eye region
        right_eye_x = self.faceMarks.part(36).x
        right_eye_y = self.faceMarks.part(38).y
        right_eye_width = self.faceMarks.part(39).x - right_eye_x
        right_eye_height = self.faceMarks.part(40).y - right_eye_y
        pos_right_eye = Rectangle(right_eye_x, right_eye_y, right_eye_width, right_eye_height)

        return pos_left_eye, pos_right_eye

    def create_eyes(self, pos_left_eye: Rectangle, pos_right_eye: Rectangle):
        self.set_left_eye(Eye(self.img, pos_left_eye))
        self.set_right_eye(Eye(self.img, pos_right_eye))

    def find_eye_vectors(self) -> cons.Point:
        if self.get_right_eye().get_pupil() and self.get_left_eye().get_pupil():
            dist_left = self.get_pos_outer_left_eye_corner().eucledian_distance(self.get_pos_inner_left_eye_corner())
            y_left_o = (self.get_left_eye().get_pupil().get_global_position_center().y -
                        self.get_pos_outer_left_eye_corner().y)
            y_left_i = (self.get_left_eye().get_pupil().get_global_position_center().y -
                        self.get_pos_inner_left_eye_corner().y)
            y_left = (y_left_i + y_left_o)/2

            x_left_o = (self.get_left_eye().get_pupil().get_global_position_center().x -
                        self.get_pos_outer_left_eye_corner().x)
            x_left_i = (self.get_left_eye().get_pupil().get_global_position_center().x -
                        self.get_pos_inner_left_eye_corner().x)
            x_left = (x_left_o+x_left_i)/2

            dist_right = self.get_pos_outer_right_eye_corner().eucledian_distance(self.get_pos_inner_right_eye_corner())
            y_right_o = (self.get_right_eye().get_pupil().get_global_position_center().y -
                         self.get_pos_outer_right_eye_corner().y)
            y_right_i = (self.get_right_eye().get_pupil().get_global_position_center().y -
                         self.get_pos_inner_right_eye_corner().y)
            y_right = (y_right_i + y_right_o) / 2

            x_right_o = (self.get_right_eye().get_pupil().get_global_position_center().x -
                         self.get_pos_outer_right_eye_corner().x)
            x_right_i = (self.get_right_eye().get_pupil().get_global_position_center().x -
                         self.get_pos_inner_right_eye_corner().x)
            x_right = (x_right_o + x_right_i) / 2
            # TODO Normalization
            return cons.Point((x_left + x_right)/2, (y_left + y_right)/2)
        return None

    # Attempt to get better y-results, but gave worse results
    # def find_eye_vectors(self) -> (cons.Point, cons.Point):
    #     # Check whether the pupils are detected
    #     if self.get_right_eye().get_pupil() and self.get_left_eye().get_pupil():
    #         if self.get_left_eye().get_pupil().get_global_roi().get_upper_left_corner().y < cons.LEFT_EYE_Y:
    #             y_left = self.get_left_eye().get_pupil().get_global_roi().get_upper_left_corner().y + \
    #                      self.get_left_eye().get_pupil().get_global_roi().get_height() - cons.LEFT_EYE_HEIGHT/2
    #         elif self.get_left_eye().get_pupil().get_global_roi().get_upper_left_corner().y > cons.LEFT_EYE_Y:
    #             y_left = self.get_left_eye().get_pupil().get_global_roi().get_upper_left_corner().y + \
    #                      cons.LEFT_EYE_HEIGHT/2
    #         else:
    #             y_left = self.get_left_eye().get_pupil().get_global_position_center().y - \
    #                      self.get_pos_left_eye_corner().y
    #
    #         if self.get_right_eye().get_pupil().get_global_roi().get_upper_left_corner().y < cons.RIGHT_EYE_Y:
    #             y_right = self.get_right_eye().get_pupil().get_global_roi().get_upper_left_corner().y + \
    #                      self.get_right_eye().get_pupil().get_global_roi().get_height() - cons.RIGHT_EYE_HEIGHT/2
    #         elif self.get_right_eye().get_pupil().get_global_roi().get_upper_left_corner().y > cons.RIGHT_EYE_Y:
    #             y_right = self.get_right_eye().get_pupil().get_global_roi().get_upper_left_corner().y + \
    #                      cons.RIGHT_EYE_HEIGHT/2
    #         else:
    #             y_right = self.get_right_eye().get_pupil().get_global_position_center().y - \
    #                      self.get_pos_right_eye_corner().y
    #
    #         x_left = self.get_left_eye().get_pupil().get_global_position_center().x - self.get_pos_left_eye_corner().x
    #         x_right = self.get_right_eye().get_pupil().get_global_position_center().x - \
    #             self.get_pos_right_eye_corner().x
    #
    #         return cons.Point(x_left, y_left), cons.Point(x_right, y_right)
    #     return None


def detect_led(face: np.ndarray) -> cons.Point:
    _, threshold = cv2.threshold(face, 245, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if x > int(w/3):
                return cons.Point(int((x+w)/2), int((y+h)/2))
    else:
        return None


class ScreenAreaBoundary(IntEnum):
    W1 = SCREEN_WIDTH
    W2 = 2 * SCREEN_WIDTH / 3
    W3 = SCREEN_WIDTH / 3
    W4 = 0

    H1 = SCREEN_HEIGHT
    H2 = 2 * SCREEN_HEIGHT / 3
    H3 = SCREEN_HEIGHT / 3
    H4 = 0


class CenterXScreenArea(IntEnum):
    L = int(SCREEN_WIDTH / 6)
    M = int(SCREEN_WIDTH / 2)
    R = int(5 * SCREEN_WIDTH / 6)


class CenterYScreenArea(IntEnum):
    T = int(SCREEN_HEIGHT / 6)
    M = int(SCREEN_HEIGHT / 2)
    D = int(5 * SCREEN_HEIGHT / 6)
