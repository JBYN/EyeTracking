import cv2

FACE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

CAM = cv2.VideoCapture(0)  # 0 used as parameter to use the webcam of the computer/laptop

SMOOTH_FACTOR = 0.005
GRADIENT_THRESHOLD = 0.3
BLUR_WEIGHT_SIZE = 5
POSTPROCESS_THRESHOLD = 0.9

COLOR_PUPIL_IND = (0, 255, 0)
RADIUS_PUPIL_IND = 3

COLOR_EYECORNER_IND = (0, 0, 255)
RADIUS_EYECORNER_IND = 1

COLOR_EYE_IND = (255, 0, 0)
