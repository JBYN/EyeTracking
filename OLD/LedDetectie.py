import cv2
import  numpy as np
import constants as cons


def main():
    while True:
        b, img = cons.CAM.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        width = gray.shape[1]
        w = int(int(width)/int(2))
        crop = gray[0:height, w: width]
        detect_led(crop)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cons.CAM.release()
    cv2.destroyAllWindows()


def detect_led(img):
    _, threshold = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=lambda x1: cv2.contourArea(x1), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            print(str(x) + ", " + str(y))
            if x != 0:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), cv2.FILLED)
    cv2.imshow("gray", img)


if __name__ == "__main__":
    main()
