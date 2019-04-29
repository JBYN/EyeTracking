# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import dlib
import model as m
import constants as cons
import view
import calibrate


def main():
    faces = list()
    while True:
        print("START")
        b, img = cons.CAM.read()
        parameters = m.EyeParameters(cons.FACE_CASCADE, cons.EYE_CASCADE, b, img)
        face = findFace(parameters)

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

                leftEyePupil = face.getLeftEye().getPupil()
                rightEyePupil = face.getRightEye().getPupil()

                #if leftEyePupil is not None and rightEyePupil is not None:
                if rightEyePupil is not None:
                    # TODO TEST updating view with mean values
                    # add face to list of faces
                    faces.append(face)
                    # if list is larger than value -> map and empty list
                    if len(faces) == cons.NUMBER_EYES:
                        pos = mapEyes2Screen(faces, calibrateP1, calibrateP2)
                        if pos is not None:
                            view.showPos(pos)
                        faces.clear()

                    view.showImage(img, face, "Face")

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    cons.CAM.release()
    cv2.destroyAllWindows()


def findFace(parameters: m.EyeParameters) -> m.Face:
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


def add2Points(point1: cons.Point, point2: cons.Point):
    return cons.Point(point1.x + point2.x, point1.y + point2.y)


def createCalibrate(face: m.Face, factor: float):
    blankScreen = cons.createBlankScreen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    p = cons.Point(int(cons.SCREEN_WIDTH * factor), int(cons.SCREEN_HEIGHT * factor))
    print("POINT: " + str(cons.SCREEN_WIDTH * factor))
    calibrateScreen = calibrate.CalibrateScreen(blankScreen, p)
    calibrateP = calibrate.Calibrate(calibrateScreen, face)
    return calibrateP


def mapEyes2Screen(faces: list, cal1: calibrate.Calibrate, cal2: calibrate.Calibrate) -> cons.Point:
    x, y = 0, 0
    # Sum of the coordinates of the different faces
    for face in faces:
        eyeVector = face.findEyeVector(face.getRightEye(), face.posRightEyeCorner)
        x += eyeVector.x
        y += eyeVector.y
    # Mean value of the coordinates
    x = x/len(faces)
    y = y/len(faces)
    p1 = cal1.getCalibrateScreen().getPositionPoint()
    p2 = cal2.getCalibrateScreen().getPositionPoint()
    print("MeanCalibrate1: " + str(cal1.getVectorX()) + ", " + str(cal1.getVectorY()))
    print("MeanCalibrate2: " + str(cal2.getVectorX()) + ", " + str(cal2.getVectorY()))
    alpha = interpolate(x, cal1.getVectorX(), cal2.getVectorX(), p1.x, p2.x)
    beta = interpolate(y, cal1.getVectorY(), cal2.getVectorY(), p1.y, p2.y)
    print("ScreenPoint: " + str(alpha) + "," + str(beta))

    # Decide which part of the screen is been looked
    center = findScreenArea(alpha, beta)
    return center


def interpolate(x: int, x1: int, x2: int, a1: int, a2: int) -> int:
    a = a1 + (x - x1)*(a2 - a1) / (x2 - x1)
    return int(a)


def findScreenArea(x: float, y: float) -> cons.Point:
    # get the x coordinate of the center of the screen area where's been looked
    if x > int(m.ScreenAreaBoundary.W1):
        return None
    elif x > int(m.ScreenAreaBoundary.W2):
        X = m.CenterXScreenArea.R
    elif x > int(m.ScreenAreaBoundary.W3):
        X = m.CenterXScreenArea.M
    elif x > int(m.ScreenAreaBoundary.W4):
        X = m.CenterXScreenArea.L
    else:
        return None

    # get the y coordinate of the center of the screen area where's been looked
    if y > m.ScreenAreaBoundary.H1:
        return None
    elif y > m.ScreenAreaBoundary.H2:
        Y = m.CenterYScreenArea.D
    elif y > m.ScreenAreaBoundary.H3:
        Y = m.CenterYScreenArea.M
    elif y > m.ScreenAreaBoundary.H4:
        Y = m.CenterYScreenArea.T
    else:
        return None

    return cons.Point(X, Y)


if __name__ == "__main__":
    main()
