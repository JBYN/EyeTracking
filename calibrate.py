# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:39:50 2019

@author: Jo
"""

import cv2
import numpy as np
import constants as cons
import model as m
import view


class CalibrateScreen:

    def __init__(self, screen: np.ndarray, pos_point: cons.Point, name: str):
        self.point = pos_point
        self.screen = screen
        self.name = name
        self.main()

    def get_position_point(self) -> cons.Point:
        return self.point

    def get_screen(self) -> np.ndarray:
        return self.screen

    def get_name(self) -> str:
        return self.name

    def main(self):
        cv2.circle(self.screen, (self.point.x, self.point.y), cons.RADIUS_CALIBRATE_POINT, cons.COLOR_CALIBRATE_POINT,
                   cons.THICKNESS_CALIBRATE_POINT)

    def close_screen(self):
        cv2.destroyWindow(self.name)


class CalibrateLightIntensity:
    def __init__(self, face: m.Face):
        self.face = face
        self.threshold = 0
        self.values_threshold = list()
        self.main()

    def get_threshold(self) -> float:
        return np.mean(self.values_threshold)

    def get_number_of_data(self) -> int:
        return len(self.values_threshold)

    def main(self):
        eye = self.face.get_right_eye().get_roi()
        self.set_threshold_light_intensity(self.threshold, cons.AREA_RATIO_THRESHOLD, eye)

    def update(self, face: m.Face):
        self.face = face
        self.main()

    def set_threshold_light_intensity(self, light_threshold: int, area_threshold: int, eye: np.ndarray):
        t = light_threshold
        area = 0
        _, threshold = cv2.threshold(eye, light_threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None:
            t += 1
            self.set_threshold_light_intensity(t, area_threshold, eye)
        else:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                break
            if (area/eye.size) >= area_threshold:
                self.values_threshold.append(t)
                return None
            else:
                t += 1
                self.set_threshold_light_intensity(t, area_threshold, eye)

        return None


class Calibrate:

    def __init__(self, calibrate_screen: CalibrateScreen, face):
        self.calibrateScreen = calibrate_screen
        self.face = face
        self.v_left_eye = list()
        self.v_right_eye = list()
        self.mean_left_eye = None
        self.mean_right_eye = None
        self.main()

    def main(self):
        view.show(self.calibrateScreen.get_screen(), self.calibrateScreen.get_name())
        self.find_vectors()

    def update_calibrate(self, face: m.Face):
        self.face = face
        if face.get_right_eye() and face.get_pos_right_eye_corner():
            if face.get_left_eye() and face.get_pos_left_eye_corner():
                self.find_vectors()
        if self.get_number_of_data() == cons.NUMBER_CALLIBRATE_DATA:
            self.mean_left_eye, self.mean_right_eye = self.calculate_mean(self.v_left_eye, self.v_right_eye)

    def get_calibrate_screen(self) -> CalibrateScreen:
        return self.calibrateScreen

    def get_mean_v_left_eye(self) -> cons.Point:
        return self.mean_left_eye

    def get_mean_v_right_eye(self) -> cons.Point:
        return self.mean_right_eye

    def get_number_of_data(self) -> int:
        return len(self.v_left_eye)

    def update_vectors(self, v_left_eye: cons.Point, v_right_eye: cons.Point):
        self.v_left_eye.append(v_left_eye)
        self.v_right_eye.append(v_right_eye)

    def find_vectors(self):
        if self.find_eye_vectors():
            v_left_eye, v_right_eye = self.find_eye_vectors()
            self.update_vectors(v_left_eye, v_right_eye)

    def find_eye_vectors(self) -> (cons.Point, cons.Point):
        return self.face.find_eye_vectors()

    def calculate_mean(self, list1: list, list2: list) -> (cons.Point, cons.Point):
        list1 = self.remove_outliers(list1)
        list2 = self.remove_outliers(list2)
        x1 = list()
        x2 = list()
        y1 = list()
        y2 = list()
        for l in list1:
            x1.append(l.x)
            y1.append(l.y)
        for l2 in list2:
            x2.append(l2.x)
            y2.append(l2.y)
        x1_mean = np.mean(x1)
        x2_mean = np.mean(x2)
        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        return cons.Point(int(x1_mean), int(y1_mean)), cons.Point(int(x2_mean), int(y2_mean))

    def remove_outliers(self, l: list) -> list:
        i = 0
        dist = list()
        for p in l:
            dist.append(p.get_distance_r0())
        l_bound, u_bound = self.calculate_bounds(sorted(dist))

        for d in dist:
            if l_bound > d or u_bound < d:
                del l[i]
            else:
                i += 1
        return l

    def close_screen(self):
        self.calibrateScreen.close_screen()

    @staticmethod
    def calculate_bounds(sorted_list: list) -> (float, float):
        q1, q3 = np.percentile(sorted_list, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return lower_bound, upper_bound
