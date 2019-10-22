import cv2
import numpy as np
import pyautogui

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
cam = cv2.VideoCapture(0)
s = cv2.resize(np.zeros((1, 1)) + 255, (SCREEN_WIDTH, SCREEN_HEIGHT))
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    cv2.imshow("test", gray)
    cv2.imshow("screen", s)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
