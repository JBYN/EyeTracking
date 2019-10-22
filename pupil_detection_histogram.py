import cv2
import glob
import numpy as np
import copy
import os
import time
from statistics import mean
import Process_data_pupil_detection as Process

# path = "Videos/Video1/Calibration/"
# eye = "LeftEyes/"


def init_image_corners(path: str, pos: int):
    # load left upper corner of the images
    upper_left_corners = np.loadtxt(path + "posEyes.csv", delimiter=';', skiprows=2)[:, [pos * 2, pos * 2 + 1]]
    upper_left_corners_x = upper_left_corners[:, 0]
    upper_left_corners_y = upper_left_corners[:, 1]
    return upper_left_corners_x, upper_left_corners_y


def init_inner_eye_corners(path: str, pos: int):
    # load coordinates inner eye corner
    inner_eye_corners = np.loadtxt(path + "eyeCorners.csv", delimiter=';', skiprows=2)[:, [pos * 4 + 2, pos * 4 + 2 + 1]]
    inner_eye_corners_x = inner_eye_corners[:, 0]
    inner_eye_corners_y = inner_eye_corners[:, 1]
    return inner_eye_corners_x, inner_eye_corners_y


def create_dir(path1: str, d1: str):
    p2 = path1 + d1
    d2 = os.path.join(p2)
    if not os.path.exists(d2):
        os.mkdir(d2)


windowClose = np.ones((5, 5), np.uint8)
windowOpen = np.ones((2, 2), np.uint8)
windowErode = np.ones((2, 2), np.uint8)


def detect_pupil_hist(path1: str, path2: str, modus: str, eye: str, pos: int, video: int) -> (list, list):
    # define list to save the vectors between the eye corner and the pupil
    data_x = list()
    data_y = list()
    vectors = list()
    vectors.append("X;Y")

    # lists to save pictures of different steps
    imgs_hist = list()
    imgs_thresh = list()
    imgs_mor = list()
    imgs_iris_region = list()
    imgs_mor_2 = list()
    imgs_thr_pupil = list()
    imgs_result = list()

    path = path1 + path2 + modus + "/"
    eye2 = eye + "/"
    inner_eye_corners_x, inner_eye_corners_y = init_inner_eye_corners(path, pos)
    upper_left_corners_x, upper_left_corners_y = init_image_corners(path, pos)

    # load images of the eyes
    img = glob.glob(path + eye2 + "*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i, 0) for i in img]

    duration = list()

    index = 0
    for i in img:
        start_time = time.time()
        # print(index)
        frame = cv2.resize(i, fx=5, fy=5, dsize=None, interpolation=cv2.INTER_LINEAR)
        pupil_frame = cv2.equalizeHist(frame)
        imgs_hist.append(pupil_frame)

        _, thresholded = cv2.threshold(pupil_frame, 50, 255, cv2.THRESH_BINARY_INV)
        imgs_thresh.append(thresholded)

        # remove parts that aren't the iris
        # mor = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, windowClose)
        mor = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, windowErode)
        imgs_mor.append(mor)
        # mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, windowOpen)

        # isolate the region of the iris
        contours, _ = cv2.findContours(mor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                f = copy.deepcopy(frame)
                f = f[y:y+h, x:x+w]
                imgs_iris_region.append(f)
                break

        # find location of the pixel with the lowest intensity in the selected region
        _, _, l, _ = cv2.minMaxLoc(f)

        # calculate the average intensity of a window around the founded pixel
        w = f[l[1]-5: l[1]+5, l[0]-5: l[0]+5]
        avg = np.mean(w)

        # Remove reflections in the original picture
        f = cv2.morphologyEx(f, cv2.MORPH_ERODE, windowErode)
        imgs_mor_2.append(f)

        # Isolate in this picture the region of interest of the pupil
        sel = f[l[1]-8: l[1]+8, l[0]-8: l[0]+8]
        _, t = cv2.threshold(sel, avg, 255, cv2.THRESH_BINARY_INV)
        imgs_thr_pupil.append(t)

        # Find the centre of gravity in the ROI
        M = cv2.moments(t)

        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = l[0]
            cy = l[1]

        # Rescale the centre to the global picture
        gx = cx + l[0] - 8 + x
        gy = cy + l[1] - 8 + y

        # Indicate the centre of the pupil
        cv2.circle(pupil_frame, (gx, gy), 2, (255, 255, 255), cv2.FILLED)
        imgs_result.append(pupil_frame)
        # cv2.rectangle(pupil_frame, (x + l[0]-8, y + l[1]-8), (x + l[0] + 8, y + l[1] + 8), (255, 255, 255))

        # create vector between eye corner and pupil and add to the list
        if inner_eye_corners_y[index] == 0 and inner_eye_corners_x[index] == 0:
            vx = 0
            vy = 0
        else:
            vx = gx / 5 + upper_left_corners_x[index] - inner_eye_corners_x[index]
            vy = gy / 5 + upper_left_corners_y[index] - inner_eye_corners_y[index]

        vectors.append(str(vx) + ";" + str(vy))
        data_x.append(vx)
        data_y.append(vy)

        end_time = time.time()
        duration.append(end_time - start_time)

        # t = cv2.resize(t, fx=10, fy=10, dsize=None, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("original", thresholded)
        cv2.imshow("hist", pupil_frame)
        # cv2.imshow("morph", mor)
        # cv2.imshow("contours", f)
        # cv2.imshow("iris_region", sel)
        # cv2.imshow("iris_region_t", t)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

        index += 1

    data_v = (data_x, data_y)
    create_dir(path + eye2, "Method_1")
    s = path + eye2 + "Method_1/"
    create_dir(s, "Histogram")
    create_dir(s, "Thresh1")
    create_dir(s, "Morphology1")
    create_dir(s, "Iris_region")
    create_dir(s, "Morphology2")
    create_dir(s, "Thresh_pupil")
    create_dir(s, "Pupil")

    # write the vectors to a file
    f = open(path + eye2 + "Method_1/vectors.csv", "w+")
    i = -1
    for v in vectors:
        f.write(str(v) + "\n")
        if i > 0:
            cv2.imwrite(path + eye2 + "Method_1/Histogram/im_" + str(i) + ".jpg", imgs_hist.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Thresh1/im_" + str(i) + ".jpg", imgs_thresh.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Morphology1/im_" + str(i) + ".jpg", imgs_mor.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Iris_region/im_" + str(i) + ".jpg", imgs_iris_region.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Morphology2/im_" + str(i) + ".jpg", imgs_mor_2.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Thresh_pupil/im_" + str(i) + ".jpg", imgs_thr_pupil.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_1/Pupil/im_" + str(i) + ".jpg", imgs_result.__getitem__(i))
        i += 1
    f.close()
    cv2.destroyAllWindows()

    r_b, data = Process.process_data(data_v, path, "pos.csv", eye, "METHOD1", modus, video, mean(duration))

    return r_b, data
