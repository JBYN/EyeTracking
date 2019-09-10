import cv2
import glob
import numpy as np
import random as rng
import copy

path = "Videos/Video1/Calibration/"

# load images of the eyes
img = glob.glob(path + "LeftEyes/*.jpg")
img.sort()
img = [cv2.imread(i, 0) for i in img]

# load left upper corner of the images
upper_left_corners = np.loadtxt(path + "posEyes.csv", delimiter=';', skiprows=2)[:, [0, 1]]
upper_left_corners_x = upper_left_corners[:, 0]
upper_left_corners_y = upper_left_corners[:, 1]

# load coordinates inner eye corner
inner_left_eye_corners = np.loadtxt(path + "eyeCorners.csv", delimiter=';', skiprows=2)[:, [2, 3]]
inner_left_eye_corners_x = inner_left_eye_corners[:, 0]
inner_left_eye_corners_y = inner_left_eye_corners[:, 1]

# define list to save the vectors between the eye corner and the pupil
vectors = list()
vectors.append("X;Y")

windowErode = np.ones((2, 2), np.uint8)

rng.seed(12345)
while True:
    frame = cv2.resize(img[0], dsize=None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    blur_frame = cv2.blur(frame, (9, 9))
    _, threshold = cv2.threshold(blur_frame, 70, 255, cv2.THRESH_BINARY_INV)
    canny_output = cv2.Canny(blur_frame, 50, 70)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):

            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(drawing, contours_poly, i, (0, 255, 255), cv2.FILLED)
            gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=255, param2=5, minRadius=0,
                                       maxRadius=50)
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                cv2.circle(frame, center, 2, (255, 255, 255), cv2.FILLED)
                # circle outline
                radius = circle[2]
                cv2.circle(gray, center, radius, (255, 255, 255))
            # cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
            #               (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            break

    cv2.imshow("frame", frame)
    cv2.imshow("blur", blur_frame)
    cv2.imshow("canny", canny_output)
    cv2.imshow('Contours', gray)
    cv2.imshow("threshold", threshold)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
cv2.destroyAllWindows()
