#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import numpy as np
import controller as detect
import calibrate
import model as c

#constants to show errors on the image
font = cv2.FONT_ITALIC
color = (0,0,255)
stroke = 2
line_type = cv2.LINE_AA

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

#Callibration
calibrate.callibrate()
#Middle borders
print("Middle: " + str(calibrate.middle[0]))
x_bordersMiddle = calibrate.get_callibrationResults(calibrate.middle[0])
y_bordersMiddle = calibrate.get_callibrationResults(calibrate.middle[1])
#Left borders
x_bordersLeft = calibrate.get_callibrationResults(calibrate.left[0])
y_bordersLeft = calibrate.get_callibrationResults(calibrate.left[1])
#Right borders
x_bordersRight = calibrate.get_callibrationResults(calibrate.right[0])
y_bordersRight = calibrate.get_callibrationResults(calibrate.right[1])
#Up borders
x_bordersUp = calibrate.get_callibrationResults(calibrate.up[0])
y_bordersUp = calibrate.get_callibrationResults(calibrate.up[1])
#Down borders
x_bordersDown = calibrate.get_callibrationResults(calibrate.down[0])
y_bordersDown  = calibrate.get_callibrationResults(calibrate.down[1])

f = open("Borders.txt","w+")
f.write("LEFT:" + str(x_bordersLeft))
f.write("MIDDLE:" + str(x_bordersMiddle))
f.write("RIGHT:" + str(x_bordersRight))
f.close()

cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop

def main() -> None:
    while True:
        print("start")
        b,img = cam.read()
        parameters = c.Eye_Parameters(face_cascade, eye_cascade, b,img)
        eyeCenters = detect.detect_eyeCenter(parameters)
        
        #detecting eye pupils
        if eyeCenters != None:
            TestLR(eyeCenters)
             
#        cv2.namedWindow("windowName", cv2.WND_PROP_FULLSCREEN)
#        cv2.setWindowProperty("windowName",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#        cv2.imshow("windowName", img)      
        
        key = cv2.waitKey(1)&0xff
        if key==ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    return None

def locateLook(pupils: c.EyePupils) -> str:
    centerRightEye = pupils.rightPupil
    if x_bordersMiddle[0] < centerRightEye.x < x_bordersMiddle[1]:
        print("Looking in the MIDDLE of the screen")
        return "MIDDLE"
    elif x_bordersLeft[0] < centerRightEye.x < x_bordersLeft[1]: 
        print("Looking to the LEFT of the screen")
        return "LEFT"
    elif x_bordersRight[0] < centerRightEye.x < x_bordersRight[1]:            
        print("Looking to the RIGHT of the screen")
        return "RIGHT"
    if y_bordersUp[0] < centerRightEye.y < y_bordersUp[1]:
        print("Looking to the UPPER screen")
    elif y_bordersDown[0] < centerRightEye.y < y_bordersDown[1]:
        print("Looking to the LOWER screen")
    else:print("Looking to the MIDDLE")
    
    return None

def TestLR(pupils: c.EyePupils)->None:
    background = np.zeros((800,1600,3),np.uint8)+255 #White background
    look = locateLook(pupils)
    cv2.namedWindow("TestScreen", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("TestScreen",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.putText(background,look,(200,200),font,1,color,stroke,line_type)
    cv2.imshow("TestScreen", background)
    return None

if __name__ == "__main__": main()

