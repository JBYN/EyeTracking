# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import dlib
import model as m
import constants as cons
import view


def main():
    while True:
        print("START")
        b, img = cons.CAM.read()
        parameters = m.EyeParameters(cons.FACE_CASCADE, cons.EYE_CASCADE, b, img)
        face = getFace(parameters)

        if face is not None:
            EYES_DETECTED = face.findEyes()
            if EYES_DETECTED:
                print("EYES detected!")
                leftEye = m.Eye(face.getROIFace(), face.getPosLeftEye())
                rightEye = m.Eye(face.getROIFace(), face.getPosRightEye())
                face.setLeftEye(leftEye)
                face.setRightEye(rightEye)

                leftEyePupil = leftEye.getPupil()
                rightEyePupil = rightEye.getPupil()

                if leftEyePupil is not None and rightEyePupil is not None:
                    print("EYECENTERS detected")
                    scalePositions(face)
                    view.showImage(img, face, "Face")

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    cons.CAM.release()
    cv2.destroyAllWindows()


# Detecting the 2 eyes on an image of a person using haarcascades.
# When the 2 eyes are detected, the region where they are detected are returned.
# Otherwise the value None will be returned
# @param face_cascade: is a CascadeClassifier which is used to detect faces
# @param eye_cascade: is a CascadeClassifier which is used to detect eyes
# @param b:
# @param img: the image where the eyes has to be detect on
# @return:
def getFace(parameters: m.EyeParameters) -> m.Face:
    # two_eyes = False  # Not 2 eyes detected yet

    # to make it possible to detect faces the capture has to be in grayscale.
    gray = cv2.cvtColor(parameters.image, cv2.COLOR_BGR2GRAY)
    faces = parameters.face_cascade.detectMultiScale(gray, 1.3, 5,
                                                     0 | cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT)

    if faces != ():
        print("FACE detected!")
        # getting the coordinates of the detected faces
        for (x, y, w, h) in faces:
            posFace = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
            face = m.Face(gray, posFace)
            return face

    else:
        print("No face detected")

    if not parameters.view:
        print("The camera is not working")
    return None


def add2Points(point1: m.Point, point2: m.Point):
    return m.Point(point1.x + point2.x, point1.y + point2.y)


def scalePositions(face: m.Face):
    p_face = m.Point(face.getPosFace().getLeftUpperCorner().x, face.getPosFace().getLeftUpperCorner().y)
    p_leftEye = m.Point(face.getPosLeftEye().getLeftUpperCorner().x, face.getPosLeftEye().getLeftUpperCorner().y)
    p_leftPupil = face.getLeftEye().getPupil().getPosEyeCenter()
    face.getLeftEye().setGlobalPositionEye(add2Points(p_face, p_leftEye))
    face.getLeftEye().getPupil().setGlobalPosition(add2Points(face.getLeftEye().getGlobalPositionEye(), p_leftPupil))

    p_rightEye = m.Point(face.getPosRightEye().getLeftUpperCorner().x, face.getPosRightEye().getLeftUpperCorner().y)
    p_rightPupil = face.getRightEye().getPupil().getPosEyeCenter()
    face.getRightEye().setGlobalPositionEye(add2Points(p_face, p_rightEye))
    face.getRightEye().getPupil().setGlobalPosition(add2Points(p_face, add2Points(p_rightEye, p_rightPupil)))


if __name__ == "__main__":
    main()
