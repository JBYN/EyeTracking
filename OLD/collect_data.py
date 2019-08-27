# source
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import controller as c
import model as m
import constants as cons
import view
import calibrate


def main():
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
            face = c.find_face(parameters)
        else:
            updated = face.update(gray)
            if updated is False:
                print("FALSE UPDATE")
                parameters = m.FaceParameters(cons.FACE_CASCADE, b, gray)
                face = c.find_face(parameters)

        if face is not None:
            prev_pos_face = True
            # Calibrate
            if not cons.CALIBRATESCREEN_P1:
                screen = create_calibrate_screen(0.5)
                view.show(screen.get_screen(), "SCREEN")
                cons.CALIBRATESCREEN_P1 = True
            else:
                if num == 5:
                    cons.LIGHT_INTENSITY_THRESHOLD += 2
                    n += 1
                    num = 0
                else:
                    num += 1
            print(str(num))
        cv2.waitKey(1) & 0xff
        if n == 50:
            num = 0
            f = open("area.csv", "w+")
            for eye in cons.eyes:
                r = cv2.resize(eye, (750, 500), cv2.INTER_AREA)
                cv2.imwrite('Eyes/' + str(num) + '.jpg', r)
                f.write(str(cons.area_pupil[num]) + '\n')
                num += 1
            break
    f.close()
    cons.CAM.release()
    cv2.destroyAllWindows()


def create_calibrate_screen(factor: float) -> calibrate.CalibrateScreen:
    blank_screen = cons.create_blank_screen(cons.SCREEN_WIDTH, cons.SCREEN_HEIGHT)
    p = cons.Point(int(cons.SCREEN_WIDTH * factor), int(cons.SCREEN_HEIGHT * factor))
    return calibrate.CalibrateScreen(blank_screen, p)


if __name__ == "__main__":
    main()
