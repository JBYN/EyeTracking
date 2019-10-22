import cv2
import glob
import numpy as np
import copy
import os
import time
from statistics import mean
import Process_data_pupil_detection as Process

# path = "Videos/Video2/Calibration/"
# eye = "LeftEyes/"

# load images of the face
# faces = glob.glob(path + "*.jpg")
# faces.sort(key=len)
# faces = [cv2.imread(face, 0) for face in faces]


def create_dir(path1: str, d1: str):
    p2 = path1 + d1
    d2 = os.path.join(p2)
    if not os.path.exists(d2):
        os.mkdir(d2)


def init_image_corners(path: str, pos: int):
    # load left upper corner of the images
    upper_left_corners = np.loadtxt(path + "posEyes.csv", delimiter=';', skiprows=2)[:, [pos * 2, pos * 2 +1]]
    upper_left_corners_x = upper_left_corners[:, 0]
    upper_left_corners_y = upper_left_corners[:, 1]
    return upper_left_corners_x, upper_left_corners_y


def init_eye_corners(path: str, pos: int):
    # load coordinates inner eye corner
    inner_eye_corners = np.loadtxt(path + "eyeCorners.csv", delimiter=';', skiprows=2)[:, [pos * 4 + 2, pos * 4 + 2 + 1]]
    inner_eye_corners_x = inner_eye_corners[:, 0]
    inner_eye_corners_y = inner_eye_corners[:, 1]
    return inner_eye_corners_x, inner_eye_corners_y


def detect_pupil(path1: str, path2: str, modus: str, eye: str, pos: int, video: int) -> (list, list):
    # define list to save the vectors between the eye corner and the pupil
    data_x = list()
    data_y = list()
    vectors = list()
    vectors.append("X;Y")

    # save the images of the different steps
    imgs_threshold = list()
    imgs_iris = list()
    imgs_pupil = list()

    path = path1 + path2 + modus + "/"
    eye2 = eye + "/"
    upper_left_corners_x, upper_left_corners_y = init_image_corners(path, pos)
    inner_eye_corners_x, inner_eye_corners_y = init_eye_corners(path, pos)

    # load images of the eyes
    img = glob.glob(path + eye2 + "*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i, 0) for i in img]

    duration = list()

    index = 0
    for im in img:
        start_time = time.time()

        # Test = faces.__getitem__(index)
        frame = cv2.resize(im, dsize=None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        f = copy.deepcopy(frame)
        # preprocess the image bu blurring using a window of 9*9 pixels
        blur_frame = cv2.blur(frame, (9, 9))

        # convert the image to a binary with the use of a predefined threshold
        _, threshold = cv2.threshold(blur_frame, 80, 255, cv2.THRESH_BINARY_INV)
        imgs_threshold.append(threshold)
        t = copy.deepcopy(threshold)
        # Detect the circles in the drawing
        circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, 1, 100, param1=255, param2=5, minRadius=0,
                                   maxRadius=50)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Only one circle should be detected, the iris
            for circle in circles[0, :]:
                # The center of the circle is considered as the position of the pupil
                center = (circle[0], circle[1])
                cv2.circle(f, center, 2, (255, 255, 255), cv2.FILLED)
                # circle outline
                radius = circle[2]
                cv2.circle(t, center, radius, (0, 255, 0))
                cv2.circle(f, center, radius, (0, 255, 0))

                if inner_eye_corners_y[index] == 0 and inner_eye_corners_x[index] == 0:
                    # Blinking eye
                    vx = 0
                    vy = 0
                else:
                    vx = center[0] / 5 + upper_left_corners_x[index] - inner_eye_corners_x[index]
                    vy = center[1] / 5 + upper_left_corners_y[index] - inner_eye_corners_y[index]

                vectors.append(str(vx) + ";" + str(vy))
                data_x.append(vx)
                data_y.append(vy)

                # gx = int(upper_left_corners_x[index] + center[0]/5)
                # gy = int(upper_left_corners_y[index] + center[1]/5)
                # cv2.circle(Test, (int(upper_left_corners_x[index]), int(upper_left_corners_y[index])), 1, (255, 255, 255),
                #            cv2.FILLED)
                # cv2.circle(Test, (gx, gy), 1, (255, 255, 255), cv2.FILLED)
                imgs_iris.append(t)
                imgs_pupil.append(f)
                break
        else:
            data_x.append(0)
            data_y.append(0)

        end_time = time.time()
        duration.append(end_time - start_time)

        cv2.imshow("frame", frame)
        cv2.imshow("blur", blur_frame)
        cv2.imshow("threshold", threshold)
        cv2.imshow("iris", t)
        cv2.imshow("pupil", f)
        # cv2.imshow("test", Test)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        index += 1

    data_v = (data_x, data_y)

    create_dir(path + eye2, "Method_2")
    s = path + eye2 + "Method_2/"
    create_dir(s, "Threshold")
    create_dir(s, "Iris")
    create_dir(s, "Pupil")

    f = open(path + eye2 + "Method_2/vectors.csv", "w+")
    i = -1
    for v in vectors:
        f.write(str(v) + "\n")
        if i > 0:
            cv2.imwrite(path + eye2 + "Method_2/Threshold/im_" + str(i) + ".jpg", imgs_threshold.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_2/Iris/im_" + str(i) + ".jpg", imgs_iris.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_2/Pupil/im_" + str(i) + ".jpg", imgs_pupil.__getitem__(i))
        i += 1
    f.close
    cv2.destroyAllWindows()

    return Process.process_data(data_v, path, "pos.csv", eye, "METHOD2", modus, video, mean(duration))
