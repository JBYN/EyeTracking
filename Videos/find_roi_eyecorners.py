import cv2
import glob
import dlib


def main():
    left_eyes = list()
    right_eyes = list()
    eye_corners = ["left out;; left in;; right out;; right in"]
    eye_corners.append("X;Y;X;Y;X;Y;X;Y")
    pos_eyes = ["left_eye;; right_eye"]
    pos_eyes.append("X;Y;X;Y")
    pos = 0
    face_cascade, predictor, img = init()
    for i in img:
        if pos == 0:
            faces = face_cascade.detectMultiScale(i, 1.3, 5, 0 | cv2.CASCADE_SCALE_IMAGE |
                                                  cv2.CASCADE_FIND_BIGGEST_OBJECT)
            pos = find_pos(faces)
        facemarks = predictor(i, pos)

        # isolate the eye regions
        right_eye = i[facemarks.part(37).y-5:facemarks.part(41).y+5, facemarks.part(36).x:facemarks.part(39).x]
        right_eye_upper_left = (facemarks.part(36).x, facemarks.part(37).y - 5)
        left_eye = i[facemarks.part(43).y-5:facemarks.part(47).y+5, facemarks.part(42).x: facemarks.part(45).x]
        left_eye_upper_left = (facemarks.part(42).x, facemarks.part(43).y - 5)

        # add the eye regions to the list
        right_eyes.append(right_eye)
        left_eyes.append(left_eye)
        pos_eyes.append(str(left_eye_upper_left[0]) + ";" + str(left_eye_upper_left[1]) + ";" +
                        str(right_eye_upper_left[0]) + ";" + str(right_eye_upper_left[1]))

        # find the different eye corners of the two eyes and add them to the list
        corners = str(facemarks.part(45).x) + ";" + str(facemarks.part(45).y) + ";" + str(facemarks.part(42).x) + ";" +\
            str(facemarks.part(42).y) + ";" + str(facemarks.part(36).x) + ";" + str(facemarks.part(36).y) + ";" + \
            str(facemarks.part(39).x) + ";" + str(facemarks.part(39).y)
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
    f = open("eyeCorners.csv", "w+")
    p = open("posEyes.csv", "w+")
    for a in eye_corners:
        f.write(str(a) + "\n")
        p.write(str(pos_eyes.__getitem__(i+2)) + "\n")
        # if i >= 0:
        #     cv2.imwrite("Video1/Calibration/LeftEyes/" + str(i) + ".jpg", left_eyes.__getitem__(i))
        #     cv2.imwrite("Video1/Calibration/RightEyes/" + str(i) + ".jpg", right_eyes.__getitem__(i))
        i += 1
    f.close()
    p.close()


def init():
    FACE_CASCADE = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    FACEMARK_PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    img = glob.glob("Video4/Calibration/*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i) for i in img]
    return FACE_CASCADE, FACEMARK_PREDICTOR, img


def find_pos(faces: list) -> dlib.rectangle:
    if faces != ():
        # getting the coordinates of the detected faces
        for (x, y, w, h) in faces:
            pos_face = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
            return pos_face


if __name__ == "__main__":
    main()
