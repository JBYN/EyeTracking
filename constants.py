import cv2
import dlib
import numpy as np
import pyautogui

FACE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

CAM = cv2.VideoCapture(0)  # 0 used as parameter to use the webcam of the computer/laptop

SMOOTH_FACTOR = 0.005
BLUR_WEIGHT_SIZE = 21
PUPIL_THRESHOLD = 40


class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def createBlankScreen(width: int, height: int):
    return cv2.resize(np.zeros((1, 1)) + 255, (width, height))


# Screen property
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


# Indicators
COLOR_PUPIL_IND = (0, 255, 0)
RADIUS_PUPIL_IND = 3

COLOR_EYECORNER_IND = (0, 0, 255)
RADIUS_EYECORNER_IND = 1

COLOR_EYE_IND = (255, 0, 0)

# Calibrate
NUMBER_CALLIBRATE_DATA = 50
RADIUS_CALIBRATE_POINT = 10
COLOR_CALIBRATE_POINT = (0, 0, 0)
THICKNESS_CALIBRATE_POINT = cv2.FILLED
CALIBRATE_P1_FACTOR = 0.125
CALIBRATE_P2_FACTOR = 0.875
CALIBRATESCREEN_P1 = False
CALIBRATESCREEN_P2 = False
NAME_CALIBRATE_WINDOW = "Calibrate"

# Indicator for looking position
RADIUS_LOOK_POINT = 10
COLOR_LOOK_POINT = (0, 0, 0)
THICKNESS_LOOK_POINT = cv2.FILLED

# Facemark
FACEMARK_PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

