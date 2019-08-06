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
        return float(np.mean(self.values_threshold))

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


class CalibratePupilSpecs:
    def __init__(self, face: m.Face):
        self.face = face
        self.values_y_left_pupil = list()
        self.values_y_right_pupil = list()
        self.heights_left_pupil = list()
        self.heights_right_pupil = list()
        self.main()

    def main(self):
        specs_left_pupil, specs_right_pupil = self.find_specs()
        self.update_lists(specs_left_pupil, specs_right_pupil)

    def update(self, face):
        self.face = face
        self.main()

    def update_lists(self, specs_left_pupil, specs_right_pupil):
        self.heights_left_pupil.append(specs_left_pupil.height)
        self.values_y_left_pupil.append(specs_left_pupil.y)
        self.heights_right_pupil.append(specs_right_pupil.height)
        self.values_y_right_pupil.append(specs_right_pupil.y)

    def get_number_of_data(self) -> int:
        return len(self.values_y_left_pupil)

    def get_y_left_pupil(self) -> float:
        return float(np.mean(self.values_y_left_pupil))

    def get_height_left_pupil(self) -> float:
        return float(np.mean(self.heights_left_pupil))

    def get_y_right_pupil(self) -> float:
        return float(np.mean(self.values_y_right_pupil))

    def get_height_right_pupil(self) -> float:
        return float(np.mean(self.heights_right_pupil))

    def find_specs(self) -> (m.Rectangle, m.Rectangle):
        right_pupil = self.face.get_right_eye().get_pupil().get_global_roi()
        left_pupil = self.face.get_left_eye().get_pupil().get_global_roi()
        return left_pupil, right_pupil


class Calibrate2points:
    def __init__(self, calibrate_screen: CalibrateScreen, face):
        self.calibrateScreen = calibrate_screen
        self.face = face
        self.eye_vector = list()
        self.mean_eye_vector = None
        self.main()

    def main(self):
        view.show(self.calibrateScreen.get_screen(), self.calibrateScreen.get_name())
        self.find_vectors()

    def update_calibrate(self, face: m.Face):
        self.face = face
        if face.get_right_eye() and face.get_pos_outer_right_eye_corner():
            if face.get_left_eye() and face.get_pos_outer_left_eye_corner():
                self.find_vectors()
        if self.get_number_of_data() == cons.NUMBER_CALLIBRATE_DATA:
            self.mean_eye_vector = self.calculate_mean(self.eye_vector)

    def get_calibrate_screen(self) -> CalibrateScreen:
        return self.calibrateScreen

    def get_mean_eye_vector(self) -> cons.Point:
        return self.mean_eye_vector

    def get_number_of_data(self) -> int:
        return len(self.eye_vector)

    def update_vectors(self, vector_eye: cons.Point):
        self.eye_vector.append(vector_eye)

    def find_vectors(self):
        if self.find_eye_vectors():
            vector = self.find_eye_vectors()
            self.update_vectors(vector)

    def find_eye_vectors(self) -> cons.Point:
        return self.face.find_eye_vectors()

    def calculate_mean(self, list1: list) -> cons.Point:
        list1 = self.remove_outliers(list1)
        x = list()
        y = list()
        # split points into x and y coordinate
        for l in list1:
            x.append(l.x)
            y.append(l.y)
        # calculate the average x and y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        return cons.Point(int(x_mean), int(y_mean))

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
