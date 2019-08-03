import cv2
import dlib
import numpy as np
import pyautogui

COLLECT_DATA = False
REFERENCE = False
TEST_SYSTEM = False

FACE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
# EYE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

CAM = cv2.VideoCapture(0)  # 0 used as parameter to use the webcam of the computer/laptop

SMOOTH_FACTOR = 0.005
BLUR_WEIGHT_SIZE = 3

LIGHT_INTENSITY_THRESHOLD = 0
AREA_RATIO_THRESHOLD = 0.18
NUMBER_EYES = 25

SIZE_EYE = 250


class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_distance_r0(self) -> float:
        return np.sqrt((np.square(self.x)+np.square(self.y)))

    @property
    def to_string(self) -> str:
        return str(self.x) + ";" + str(self.y)


def create_blank_screen(width: int, height: int):
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
NUMBER_CALLIBRATE_DATA = 100
RADIUS_CALIBRATE_POINT = 10
COLOR_CALIBRATE_POINT = (0, 0, 0)
THICKNESS_CALIBRATE_POINT = cv2.FILLED
CALIBRATE_P1_FACTOR = 0.125  # Factor to place the first case's mark
CALIBRATE_P2_FACTOR = 0.875  # Factor to place the second case's mark
CALIBRATESCREEN_P1 = False  # Boolean to indicate the status of the first case's calibrating screen
CALIBRATESCREEN_P2 = False  # Boolean to indicate the status of the second case's calibrating screen
NAME_CALIBRATE_WINDOW = "Calibrate"

# Indicator for looking position
RADIUS_LOOK_POINT = 10
COLOR_LOOK_POINT = (0, 0, 0)
THICKNESS_LOOK_POINT = cv2.FILLED

# Facemark
FACEMARK_PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

# DEBUG
threshold_value = list()
threshold = list()
eyes = list()
area_pupil = list()
size_eye = list()

led_position = list()
pupil_position = list()
corner_position = list()
