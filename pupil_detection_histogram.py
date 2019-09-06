import cv2
import glob
import numpy as np
import copy

img = glob.glob("Videos/Video1/Calibration/LeftEyes/*.jpg")
img.sort()
img = [cv2.imread(i, 0) for i in img]

windowClose = np.ones((5, 5), np.uint8)
windowOpen = np.ones((2, 2), np.uint8)
windowErode = np.ones((2, 2), np.uint8)

for i in img:
    frame = cv2.resize(i, fx=5, fy=5, dsize=None, interpolation=cv2.INTER_LINEAR)
    pupil_frame = cv2.equalizeHist(frame)
    _, thresholded = cv2.threshold(pupil_frame, 50, 255, cv2.THRESH_BINARY_INV)

    # remove parts that aren't the iris
    # mor = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, windowClose)
    mor = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, windowErode)
    # mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, windowOpen)

    # isolate the region of the iris
    contours, _ = cv2.findContours(mor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            f = copy.deepcopy(frame)
            f = f[y:y+h, x:x+w]
            break

    # find location of the pixel with the lowest intensity in the selected region
    _, _, l, _ = cv2.minMaxLoc(f)

    # calculate the average intensity of a window around the founded pixel
    w = f[l[1]-5: l[1]+5, l[0]-5: l[0]+5]
    avg = np.mean(w)

    # Remove reflections in the original picture
    f = cv2.morphologyEx(f, cv2.MORPH_ERODE, windowErode)

    # Isolate in this picture the region of interest of the pupil
    sel = f[l[1]-8: l[1]+8, l[0]-8: l[0]+8]
    _, t = cv2.threshold(sel, avg, 255, cv2.THRESH_BINARY_INV)

    # Find the centre of gravity in the ROI
    M = cv2.moments(t)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Rescale the centre to the global picture
    gx = cx + l[0] + x - 8
    gy = cy + l[1] + y - 8

    # Indicate the centre of the pupil
    cv2.circle(pupil_frame, (gx, gy), 2, (255, 255, 255), cv2.FILLED)
    # cv2.rectangle(pupil_frame, (x + l[0]-8, y + l[1]-8), (x + l[0] + 8, y + l[1] + 8), (255, 255, 255))

    # t = cv2.resize(t, fx=10, fy=10, dsize=None, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("original", thresholded)
    cv2.imshow("hist", pupil_frame)
    # cv2.imshow("morph", mor)
    # cv2.imshow("contours", f)
    # cv2.imshow("iris_region", sel)
    # cv2.imshow("iris_region_t", t)
    key = cv2.waitKey(1000) & 0xff
    if key == ord('q'):
        break
cv2.destroyAllWindows()
