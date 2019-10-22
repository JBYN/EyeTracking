import numpy as np
import cv2
import glob
import dlib

nr_video = "16"
video = "Video" + nr_video + "/"
modus = "Calibration/"
path = video + modus


def main():
    left_eyes = list()
    right_eyes = list()
    eye_corners = ["left out;; left in;; right out;; right in"]
    eye_corners.append("X;Y;X;Y;X;Y;X;Y")
    pos_eyes = ["left_eye;; right_eye"]
    pos_eyes.append("X;Y;X;Y")
    pos = 0
    face_cascade, predictor, img = init()
    index = 0
    for i in img:
        # mirror it
        i = cv2.flip(i, 1)
        if pos == 0:
            faces = face_cascade.detectMultiScale(i, 1.3, 5, 0 | cv2.CASCADE_SCALE_IMAGE |
                                                  cv2.CASCADE_FIND_BIGGEST_OBJECT)
            pos = find_pos(faces)
        facemarks = predictor(i, pos)

        # isolate the eye regions
        left_eye = i[facemarks.part(37).y-5:facemarks.part(41).y+5, facemarks.part(36).x:facemarks.part(39).x]
        left_eye_upper_left = (facemarks.part(36).x, facemarks.part(37).y - 5)
        right_eye = i[facemarks.part(43).y-5:facemarks.part(47).y+5, facemarks.part(42).x: facemarks.part(45).x]
        right_eye_upper_left = (facemarks.part(42).x, facemarks.part(43).y - 5)

        # add the eye regions to the list
        right_eyes.append(right_eye)
        left_eyes.append(left_eye)
        pos_eyes.append(str(left_eye_upper_left[0]) + ";" + str(left_eye_upper_left[1]) + ";" +
                        str(right_eye_upper_left[0]) + ";" + str(right_eye_upper_left[1]))

        # find the different eye corners of the two eyes and add them to the list
        # Check the eye for blinking, if so add coordinates (0,0) to the list corners
        ear_left = eye_aspect_ratio((facemarks.part(39).x, facemarks.part(39).y),
                                    (facemarks.part(38).x, facemarks.part(38).y),
                                    (facemarks.part(37).x, facemarks.part(37).y),
                                    (facemarks.part(36).x, facemarks.part(36).y),
                                    (facemarks.part(41).x, facemarks.part(41).y),
                                    (facemarks.part(40).x, facemarks.part(40).y))

        if ear_left < 0.1:
            left_corners = str(0) + ";" + str(0) + ";" + str(0) + ";" + str(0)
        else:
            left_corners = str(facemarks.part(36).x) + ";" + str(facemarks.part(36).y) + ";" + \
                           str(facemarks.part(39).x) + ";" + str(facemarks.part(39).y)

        ear_right = eye_aspect_ratio((facemarks.part(42).x, facemarks.part(42).y),
                                     (facemarks.part(43).x, facemarks.part(43).y),
                                     (facemarks.part(44).x, facemarks.part(44).y),
                                     (facemarks.part(45).x, facemarks.part(45).y),
                                     (facemarks.part(46).x, facemarks.part(46).y),
                                     (facemarks.part(47).x, facemarks.part(47).y))

        if ear_right < 0.1:
            right_corners = str(0) + ";" + str(0) + ";" + str(0) + ";" + str(0)
        else:
            right_corners = str(facemarks.part(45).x) + ";" + str(facemarks.part(45).y) + ";" + \
                            str(facemarks.part(42).x) + ";" + str(facemarks.part(42).y)

        corners = left_corners + ";" + right_corners
        eye_corners.append(corners)

        # right_eye = cv2.resize(right_eye, fx=5, fy=5, dsize=None, interpolation=cv2.INTER_LINEAR)
        # left_eye = cv2.resize(left_eye, fx=5, fy=5, dsize=None, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("re", right_eye)
        # cv2.imshow("le", left_eye)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    # write data to files
    i = -2
    f = open(path + "eyeCorners.csv", "w+")
    p = open(path + "posEyes.csv", "w+")
    for a in eye_corners:
        f.write(str(a) + "\n")
        p.write(str(pos_eyes.__getitem__(i+2)) + "\n")
        if i >= 0:
            cv2.imwrite(path + "LeftEyes/" + str(i) + ".jpg", left_eyes.__getitem__(i))
            cv2.imwrite(path + "RightEyes/" + str(i) + ".jpg", right_eyes.__getitem__(i))
        i += 1
    f.close()
    p.close()


def init():
    FACE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    FACEMARK_PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    img = glob.glob(path + "*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i) for i in img]
    return FACE_CASCADE, FACEMARK_PREDICTOR, img


def find_pos(faces: list) -> dlib.rectangle:
    if faces != ():
        # getting the coordinates of the detected faces
        for (x, y, w, h) in faces:
            pos_face = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
            return pos_face


def eye_aspect_ratio(p1: tuple, p2: tuple, p3: tuple, p4: tuple, p5: tuple, p6: tuple) -> float:
    return (orthogonal_distance(p2, p6) + orthogonal_distance(p3, p5)) / (2 * orthogonal_distance(p1, p4))


def orthogonal_distance(p1: tuple, p2: tuple) -> float:
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


if __name__ == "__main__":
    main()
