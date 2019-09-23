import cv2
import glob
import numpy as np
import copy

path = "Videos/Video1/Calibration/"

# load images of the eyes
img = glob.glob(path + "LeftEyes/*.jpg")
img.sort(key=len)
img = [cv2.imread(i, 0) for i in img]

# load images of the face
# faces = glob.glob(path + "*.jpg")
# faces.sort(key=len)
# faces = [cv2.imread(face, 0) for face in faces]

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

# save the images of the different steps
imgs_threshold = list()
imgs_iris = list()
imgs_pupil = list()

index = 0
for im in img:
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

        vx = center[0]/5 + upper_left_corners_x[index] - inner_left_eye_corners_x[index]
        vy = center[1]/5 + upper_left_corners_y[index] - inner_left_eye_corners_y[index]
        vectors.append(str(vx) + ";" + str(vy))

        # gx = int(upper_left_corners_x[index] + center[0]/5)
        # gy = int(upper_left_corners_y[index] + center[1]/5)
        # cv2.circle(Test, (int(upper_left_corners_x[index]), int(upper_left_corners_y[index])), 1, (255, 255, 255),
        #            cv2.FILLED)
        # cv2.circle(Test, (gx, gy), 1, (255, 255, 255), cv2.FILLED)
        imgs_iris.append(t)
        imgs_pupil.append(f)
        break

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

f = open(path + "LeftEyes/Method_2/vectors.csv", "w+")
i = -1
for v in vectors:
    f.write(str(v) + "\n")
    if i > 0:
        cv2.imwrite(path + "LeftEyes/Method_2/Threshold/im_" + str(i) + ".jpg", imgs_threshold.__getitem__(i))
        cv2.imwrite(path + "LeftEyes/Method_2/Iris/im_" + str(i) + ".jpg", imgs_iris.__getitem__(i))
        cv2.imwrite(path + "LeftEyes/Method_2/Pupil/im_" + str(i) + ".jpg", imgs_pupil.__getitem__(i))
    i += 1
f.close
cv2.destroyAllWindows()

