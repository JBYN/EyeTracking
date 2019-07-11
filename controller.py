# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import dlib
import model as m
import constants as cons
import view
import calibrate


def main():
    if cons.COLLECT_DATA:
        collect_data()
        return 0

    faces = list()
    prev_pos_face = False
    while True:
        print("START")
        b, img = cons.CAM.read()
        # to make it possible to detect faces the capture has to be in grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_pos_face is False:
            parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
            face = find_face(parameters)
        else:
            Updated = face.update(gray)
            if Updated is False:
                print("FALSE UPDATE")
                parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
                face = find_face(parameters)

        if face is not None:
            prev_pos_face = True
            # Calibrate
            if not cons.CALIBRATESCREEN_P1:
                # Set threshold for the light intensity
                screen = createCalibrateScreen(0.5)
                T = calibrate.CalibrateLightIntensity(face, screen)
                cons.PUPIL_THRESHOLD = T.getThreshold()

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
                if calibrateP1.getVectorY() == calibrateP2.getVectorY():
                    cons.CALIBRATESCREEN_P1 = False
                    calibrateP1 = None
                    cons.CALIBRATESCREEN_P2 = False
                    calibrateP2 = None
                else:
                    leftEyePupil = face.getLeftEye().get_pupil()
                    rightEyePupil = face.getRightEye().get_pupil()

                    if rightEyePupil is not None:
                        # TODO TEST updating view with mean values
                        # add face to list of faces
                        faces.append(face)
                        # if list is larger than value -> map and empty list
                        if len(faces) == cons.NUMBER_EYES:
                            pos = mapEyes2Screen(faces, calibrateP1, calibrateP2)
                            if pos is not None:
                                view.showPos(pos)
                                view.showImage(img, face, "Face")
                            faces.clear()

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cons.CAM.release()
    cv2.destroyAllWindows()


def find_face(parameters: m.FaceParameters) -> m.Face:

    faces = parameters.face_cascade.detectMultiScale(parameters.image, 1.3, 5,
                                                     0 | cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT)

    if faces != ():
        print("FACE detected!")
        # getting the coordinates of the detected faces
        for (x, y, w, h) in faces:
            pos_face = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
            face = m.Face(parameters.image, pos_face)
            return face

    else:
        print("No face detected")

    if not parameters.view:
        print("The camera is not working")
    return None


def add2Points(point1: cons.Point, point2: cons.Point):
    return cons.Point(point1.x + point2.x, point1.y + point2.y)


def createCalibrateScreen(factor: float) -> calibrate.CalibrateScreen:
    blankScreen = cons.createBlankScreen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    p = cons.Point(int(cons.SCREEN_WIDTH * factor), int(cons.SCREEN_HEIGHT * factor))
    return calibrate.CalibrateScreen(blankScreen, p)


def createCalibrate(face: m.Face, factor: float):
    calibrateScreen = createCalibrateScreen(factor)
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

########################################################################################################################
####################### CODE TO COLLECT DATA TO DETERMINE "PUPIL_TRHESHOLD" ############################################


def collect_data():
    prev_pos_face = False
    num = 0
    n = 1
    while True:
        print("START")
        b, img = cons.CAM.read()
        # to make it possible to detect faces the capture has to be in grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_pos_face is False:
            parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
            face = find_face(parameters)
        else:
            updated = face.update(gray)
            if updated is False:
                print("FALSE UPDATE")
                parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
                face = find_face(parameters)

        if face is not None:
            prev_pos_face = True
            # Calibrate
            if not cons.CALIBRATESCREEN_P1:
                screen = create_calibrate_screen(0.5)
                view.show(screen.getScreen(), "SCREEN")
                cons.CALIBRATESCREEN_P1 = True
            else:
                if num == 3:
                    cons.PUPIL_THRESHOLD += 2
                    n += 1
                    num = 0
                else:
                    num += 1
            print(str(num))
        cv2.waitKey(1) & 0xff
        if n == 75:
            # Todo collect images to set light intensity threshold
            num = 0
            f = open("area.csv", "w+")
            for eye in cons.eyes:
                r = cv2.resize(eye, (750, 500), cv2.INTER_AREA)
                cv2.imwrite('Eyes/' + str(num) + '.jpg', r)
                f.write(str(cons.area[num]) + '\n')
                num += 1
            break
    f.close()
    cons.CAM.release()
    cv2.destroyAllWindows()


def create_calibrate_screen(factor: float) -> calibrate.CalibrateScreen:
    blank_screen = cons.createBlankScreen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    p = cons.Point(int(cons.SCREEN_WIDTH * factor), int(cons.SCREEN_HEIGHT * factor))
    return calibrate.CalibrateScreen(blank_screen, p)

########################################################################################################################


if __name__ == "__main__":
    main()
