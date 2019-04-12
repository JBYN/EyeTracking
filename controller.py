# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import dlib
import numpy as np
import constants
import model as m
import constants as cons
import view
import calibrate


def main():
    while True:
        print("START")
        b, img = cons.CAM.read()
        parameters = m.EyeParameters(cons.FACE_CASCADE, cons.EYE_CASCADE, b, img)
        face = getFace(parameters)

        if face is not None:
            # Calibrate
            if not cons.CALIBRATESCREEN_P1:
                calibrateP1 = createCalibrate(face, cons.CALIBRATE_P1_FACTOR)
                cons.CALIBRATESCREEN_P1 = True
            elif calibrateP1.numberOfCalibrateData < cons.NUMBER_CALLIBRATE_DATA:
                calibrateP1.updateCalibrate(face)
            elif not cons.CALIBRATESCREEN_P2:
                cv2.destroyWindow(cons.NAME_CALIBRATE_WINDOW)
                calibrateP2 = createCalibrate(face, cons.CALIBRATE_P2_FACTOR)
                cons.CALIBRATESCREEN_P2 = True
            elif calibrateP2.numberOfCalibrateData < cons.NUMBER_CALLIBRATE_DATA:
                calibrateP2.updateCalibrate(face)
            else:
                cv2.destroyWindow(cons.NAME_CALIBRATE_WINDOW)
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
                        pos = mapEyes2Screen(face, calibrateP1, calibrateP2)
                        view.showPos(pos)
                        scalePositions(face)
                        view.showImage(img, face, "Face")

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    cons.CAM.release()
    cv2.destroyAllWindows()


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


def add2Points(point1: constants.Point, point2: constants.Point):
    return constants.Point(point1.x + point2.x, point1.y + point2.y)


def scalePositions(face: m.Face):
    p_face = constants.Point(face.getPosFace().getLeftUpperCorner().x, face.getPosFace().getLeftUpperCorner().y)
    p_leftEye = constants.Point(face.getPosLeftEye().getLeftUpperCorner().x,
                                face.getPosLeftEye().getLeftUpperCorner().y)
    p_leftPupil = face.getLeftEye().getPupil().getPosEyeCenter()
    face.getLeftEye().setGlobalPositionEye(add2Points(p_face, p_leftEye))
    face.getLeftEye().getPupil().setGlobalPosition(add2Points(face.getLeftEye().getGlobalPositionEye(), p_leftPupil))

    p_rightEye = constants.Point(face.getPosRightEye().getLeftUpperCorner().x,
                                 face.getPosRightEye().getLeftUpperCorner().y)
    p_rightPupil = face.getRightEye().getPupil().getPosEyeCenter()
    face.getRightEye().setGlobalPositionEye(add2Points(p_face, p_rightEye))
    face.getRightEye().getPupil().setGlobalPosition(add2Points(p_face, add2Points(p_rightEye, p_rightPupil)))


def createCalibrate(face: m.Face, factor: float):
    blankScreen = cons.createBlankScreen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    print("WIDTH: " + str(cons.SCREEN_WIDTH))
    print("HEIGHT: " + str(cons.SCREEN_HEIGHT))
    p = constants.Point(int(cons.SCREEN_WIDTH * factor), int(cons.SCREEN_HEIGHT * factor))
    print("POINT: " + str(cons.SCREEN_WIDTH * factor))
    calibrateScreen = calibrate.CalibrateScreen(blankScreen, p)
    calibrateP = calibrate.Calibrate(calibrateScreen, face)
    return calibrateP


def mapEyes2Screen(face: m.Face, cal1: calibrate.Calibrate, cal2: calibrate.Calibrate) -> cons.Point:
    eyeCoordinates = face.findEyeVector(face.getRightEye(), face.getEyeCorner())
    p1 = cal1.getCalibrateScreen().getPositionPoint()
    p2 = cal2.getCalibrateScreen().getPositionPoint()
    alpha = interpolate(eyeCoordinates.x, cal1.getVectorX(), cal2.getVectorX(), p1.x, p2.x)
    beta = interpolate(eyeCoordinates.y, cal1.getVectorY(), cal2.getVectorY(), p1.y, p2.y)
    print("ScreenPoint: " + str(alpha) + "," + str(beta))
    return cons.Point(alpha, beta)


def interpolate(x: int, x1: int, x2: int, a1: int, a2: int) -> int:
    a = a1 + (x - x1)*(a2 - a1) / (x2 - x1)
    return int(a)


if __name__ == "__main__":
    main()
