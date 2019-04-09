#https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv-p-1/

import dlib
import numpy as np
import classes as c


class EyeCorner():

    def __init__(self, img: np.ndarray, p_face: np.ndarray):
        self.img = img
        self.posFace = p_face
        predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        self.faceMarks = predictor(self.img, p_face)

    def getPosLeftEyeCorner(self) -> c.Point:
        return c.Point(self.faceMarks.part(45).x, self.faceMarks.part(45).y)

    def getPosRightEyeCorner(self) -> c.Point:
        return c.Point(self.faceMarks.part(36).x, self.faceMarks.part(36).y)





