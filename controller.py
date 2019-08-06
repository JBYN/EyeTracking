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
    test_system = list()
    test_case = 0
    enough_data_light_intensity = False  # Boolean to indicate the end of the calibration of the light threshold.
    cal_parameters = False  # Boolean to indicate whether calibration is started.
    screen_bool_test_system = False
    screen_parameters = False  # Boolean to indicate whether the calibrating screen for the light threshold is showed.
    screen_bool = False
    prev_face = False  # Boolean to indicate whether a face was already found are not.
    print("START")
    while True:
        b, img = cons.CAM.read()
        # to make it possible to detect faces the capture has to be in gray scale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # If there isn't detected a face already
        if prev_face is False:
            # Set the parameters to detect a face
            parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
            # Find a face
            face = find_face(parameters)
        else:
            # Try to update the information of the face
            updated = face.update(gray)
            # If the update isn't successful, look for the face again
            if updated is False:
                parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
                face = find_face(parameters)

        if face:
            prev_face = True
            # Check whether the two eyes exist/are found
            if face.get_right_eye() and face.get_left_eye():
                print("EYES detected")
                # Calibrate
                # First calibrate to know the light intensity
                if enough_data_light_intensity is False:
                    if screen_parameters is False:
                        screen = create_calibrate_screen("calibrate light intensity", 0.5)
                        # show the calibrate screen
                        view.show(screen.get_screen(), screen.get_name())
                        screen_parameters = True
                    # If calibration isn't started make a calibrating object
                    elif cal_parameters is False:
                        t = calibrate.CalibrateLightIntensity(face)
                        cal_parameters = True
                    else:
                        # update the calibrating object
                        t.update(face)
                        # If enough data is collected:
                        #           Set the "LIGHT_INTENSITY_THRESHOLD"
                        #           Close the window with the calibration screen
                        if t.get_number_of_data() > 50:
                            enough_data_light_intensity = True
                            cons.LIGHT_INTENSITY_THRESHOLD = t.get_threshold()
                            screen.close_screen()
                            screen_parameters = False
                            cal_parameters = False
                # Case to collect data with a LED as reference point to compare
                # the data of the eye corner and pupil with
                elif cons.REFERENCE:
                    if screen_bool is False:
                        screen = create_calibrate_screen("Collect_data_reference", 0.5)
                        view.show(screen.get_screen(), screen.get_name())
                        screen_bool = True
                    elif len(cons.led_position) == 500:
                        screen.close_screen()
                        cons.CAM.release()
                        write_data2file_reference()
                        return 0
                # When none of the cases above start the calibration of the pupil positions
                else:
                    if not cons.CALIBRATESCREEN_P1:
                        calibrate_p1 = create_calibrate(face, cons.CALIBRATE_P1_FACTOR, cons.NAME_CALIBRATE_WINDOW)
                        cons.CALIBRATESCREEN_P1 = True
                    elif calibrate_p1.get_mean_eye_vector() is None:
                        calibrate_p1.update_calibrate(face)
                    elif not cons.CALIBRATESCREEN_P2:
                        calibrate_p1.close_screen()
                        calibrate_p2 = create_calibrate(face, cons.CALIBRATE_P2_FACTOR, cons.NAME_CALIBRATE_WINDOW)
                        cons.CALIBRATESCREEN_P2 = True
                    elif calibrate_p2.get_mean_eye_vector() is None:
                        calibrate_p2.update_calibrate(face)
                    else:
                        calibrate_p2.close_screen()

                        if int(calibrate_p1.get_mean_eye_vector().x) == int(calibrate_p2.get_mean_eye_vector().x)\
                                or int(calibrate_p1.get_mean_eye_vector().y) == \
                                int(calibrate_p2.get_mean_eye_vector().y):
                            cons.CALIBRATESCREEN_P1 = False
                            calibrate_p1 = None
                            cons.CALIBRATESCREEN_P2 = False
                            calibrate_p2 = None
                        else:
                            left_eye_pupil = face.get_left_eye().get_pupil()
                            right_eye_pupil = face.get_right_eye().get_pupil()

                            if right_eye_pupil and left_eye_pupil:
                                if cons.TEST_SYSTEM:
                                    if test_case < 9:
                                        print(len(test_system))
                                        if len(test_system) < 50 * (test_case + 1):
                                            if screen_bool_test_system is False:
                                                x_factor = 0.875 - (test_case % 3)*0.375
                                                y_factor = 0.125*((test_case // 3)*3+1)
                                                screen = create_calibrate_screen("TEST", x_factor, y_factor)
                                                view.show(screen.get_screen(), screen.get_name())
                                                screen_bool_test_system = True
                                            else:
                                                faces.append(face)
                                                area, x, y = map_eyes2screen(faces, calibrate_p1, calibrate_p2)
                                                if area:
                                                    test_system.append(str((0.875 -
                                                                            (test_case % 3)*0.375)*cons.SCREEN_WIDTH) +
                                                                       ";" + str((0.125*((test_case // 3)*3+1)) *
                                                                                 cons.SCREEN_HEIGHT) + ";" + str(x) +
                                                                       ";" + str(y) + ";" + area.to_string)
                                                faces.clear()
                                        else:
                                            test_case += 1
                                            screen_bool_test_system = False
                                    else:
                                        cons.TEST_SYSTEM = False
                                        screen.close_screen()
                                        cons.CAM.release()
                                        write_data2file_test(test_system)
                                        return 0
                                else:
                                    # add face to list of faces
                                    faces.append(face)
                                    # if list is larger than value -> map and empty list
                                    if len(faces) == cons.NUMBER_EYES:
                                        pos, _, _ = map_eyes2screen(faces, calibrate_p1, calibrate_p2)
                                        if pos:
                                            view.show_pos(pos)
                                            # view.show_image(img, face, "Face")
                                        faces.clear()
        else:
            prev_face = False

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


def create_calibrate_screen(name: str, factor_x: float, factor_y=None) -> calibrate.CalibrateScreen:
    blank_screen = cons.create_blank_screen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    if factor_y:
        p = cons.Point(int(cons.SCREEN_WIDTH * factor_x), int(cons.SCREEN_HEIGHT * factor_y))
    else:
        p = cons.Point(int(cons.SCREEN_WIDTH * factor_x), int(cons.SCREEN_HEIGHT * factor_x))
    return calibrate.CalibrateScreen(blank_screen, p, name)


def create_calibrate(face: m.Face, factor: float, name: str):
    calibrate_screen = create_calibrate_screen(name, factor)
    calibrate_p = calibrate.Calibrate2points(calibrate_screen, face)
    return calibrate_p


def map_eyes2screen(faces: list, cal1: calibrate.Calibrate2points, cal2: calibrate.Calibrate2points) -> \
        (cons.Point, int, int):
    x, y = 0, 0
    # Sum of the coordinates of the different faces
    for face in faces:
        vector = face.find_eye_vectors()

        x += vector.x
        y += vector.y

    # Mean value of the coordinates
    x_mean = x/len(faces)
    y_mean = y/len(faces)

    # position of the point on the screen during the calibration
    p1 = cal1.get_calibrate_screen().get_position_point()
    p2 = cal2.get_calibrate_screen().get_position_point()

    # mean position for the two eyes
    alpha = interpolate(x_mean, cal1.get_mean_eye_vector().x, cal2.get_mean_eye_vector().x, p1.x, p2.x)
    beta = interpolate(y_mean, cal1.get_mean_eye_vector().y, cal2.get_mean_eye_vector().y, p1.y, p2.y)

    # Decide which part of the screen is been looked
    center = find_screen_area(alpha, beta)
    return center, alpha, beta


def interpolate(x: float, x1: int, x2: int, a1: int, a2: int) -> float:
    a = a1 + (x - x1)*(a2 - a1) / (x2 - x1)
    return a


def find_screen_area(x: float, y: float) -> cons.Point:
    # get the x coordinate of the center of the screen area where's been looked
    if x > int(m.ScreenAreaBoundary.W1):
        return None
    elif x > int(m.ScreenAreaBoundary.W2):
        x_n = m.CenterXScreenArea.R
    elif x > int(m.ScreenAreaBoundary.W3):
        x_n = m.CenterXScreenArea.M
    elif x > int(m.ScreenAreaBoundary.W4):
        x_n = m.CenterXScreenArea.L
    else:
        return None

    # get the y coordinate of the center of the screen area where's been looked
    if y > m.ScreenAreaBoundary.H1:
        return None
    elif y > m.ScreenAreaBoundary.H2:
        y_n = m.CenterYScreenArea.D
    elif y > m.ScreenAreaBoundary.H3:
        y_n = m.CenterYScreenArea.M
    elif y > m.ScreenAreaBoundary.H4:
        y_n = m.CenterYScreenArea.T
    else:
        return None

    return cons.Point(x_n, y_n)


########################################################################################################################
####################### CODE TO COLLECT DATA TO DETERMINE "PUPIL_TRHESHOLD" ############################################


def collect_data():
    prev_pos_face = False
    case = 0
    num = 0
    n = 1
    while case < 3:
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
                parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
                face = find_face(parameters)

        # Check whether a previous face exists
        if face:
            prev_pos_face = True
        else:
            prev_pos_face = False

        # Show the different calibrate screens
        if not cons.CALIBRATESCREEN_P1:
            screen = create_calibrate_screen("Pupil Threshold", 0.125*(1+3*case))
            view.show(screen.get_screen(), "SCREEN")
            cons.CALIBRATESCREEN_P1 = True
            num, n = count_data(num, n)
        else:
            num, n = count_data(num, n)
        print(str(num))
        cv2.waitKey(1) & 0xff
        if n == 50:
            write_data2file("area_" + str(case) + ".csv", "Eyes_" + str(case)+"/")
            case += 1
            n = 0
    cons.CAM.release()
    cv2.destroyAllWindows()


def write_data2file(file_name_area: str, folder_name_image: str):
    num = 0
    f = open("Eyes/" + file_name_area, "w+")
    for eye in cons.eyes:
        r = cv2.resize(eye, (750, 500), cv2.INTER_AREA)
        cv2.imwrite("Eyes/" + folder_name_image + str(num) + "_" + str(cons.threshold_value[num])+'.jpg', r)
        f.write(str(cons.area_pupil[num]) + "," + str(cons.size_eye[num]) + "," + str(cons.threshold_value[num]) + '\n')
        num += 1
    f.close()
    cons.CALIBRATESCREEN_P1 = False
    rm_data_from_lists()


def rm_data_from_lists():
    cons.threshold.clear()
    cons.threshold_value.clear()
    cons.eyes.clear()
    cons.area_pupil.clear()
    cons.size_eye.clear()
    cons.LIGHT_INTENSITY_THRESHOLD = 0


def count_data(num: int, n: int):
    if num == 2:
        cons.LIGHT_INTENSITY_THRESHOLD += 2
        n += 1
        num = 0
    else:
        num += 1
    return num, n

########################################################################################################################
############### CODE TO COLLECT DATA TO DETERMINE "PUPIL_VAR" ##########################################################


def write_data2file_reference():
    num = 0
    f = open("POS/Reference.csv", "w+")
    for pos in cons.led_position:
        f.write(pos + ";" + cons.corner_position[num] + ";" + cons.pupil_position[num] + '\n')
        num += 1
    f.close()
    rm_data_from_lists_reference()


def rm_data_from_lists_reference():
    cons.led_position.clear()
    cons.pupil_position.clear()
    cons.corner_position.clear()

########################################################################################################################


def write_data2file_test(l: list):
    num = 0
    f = open("MATLAB/TEST/values.csv", "w+")
    for e in l:
        f.write(e + '\n')
        num += 1
    f.close()


if __name__ == "__main__":
    main()
