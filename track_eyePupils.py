#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import findEyeCenter as detect
import callibrating
import classes as c

#constants to show errors on the image
font = cv2.FONT_ITALIC
color = (0,0,255)
stroke = 2
line_type = cv2.LINE_AA

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

#Callibration
callibrating.callibrate()
#Middle borders
print("Middle: " + str(callibrating.middle[0]))
x_bordersMiddle = callibrating.get_callibrationResults(callibrating.middle[0])
y_bordersMiddle = callibrating.get_callibrationResults(callibrating.middle[1])
#Left borders
x_bordersLeft = callibrating.get_callibrationResults(callibrating.left[0])
y_bordersLeft = callibrating.get_callibrationResults(callibrating.left[1])
#Right borders
x_bordersRight = callibrating.get_callibrationResults(callibrating.right[0])
y_bordersRight = callibrating.get_callibrationResults(callibrating.right[1])
#Up borders
x_bordersUp = callibrating.get_callibrationResults(callibrating.up[0])
y_bordersUp = callibrating.get_callibrationResults(callibrating.up[1])
#Down borders
x_bordersDown = callibrating.get_callibrationResults(callibrating.down[0])
y_bordersDown  = callibrating.get_callibrationResults(callibrating.down[1])

#screen = np.zeros((800,1600,3),np.uint8)
cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop

while True:
    print("start")
    b,img = cam.read()
    parameters = c.Eye_Parameters(face_cascade, eye_cascade, b,img)
    eyeCenters = detect.detect_eyeCenter(parameters)
    
    #detecting eye pupils
    if eyeCenters != None:
        print("EYECENTERS detected!")
        centerRightEye = eyeCenters.rightPupil
        if x_bordersMiddle[0] < centerRightEye.x < x_bordersMiddle[1]:
            print("Looking in the MIDDLE of the screen")
        elif x_bordersLeft[0] < centerRightEye.x < x_bordersLeft[1]: 
            print("Looking to the LEFT of the screen")
        elif x_bordersRight[0] < centerRightEye.x < x_bordersRight[1]:            
            print("Looking to the RIGHT of the screen")
        if y_bordersUp[0] < centerRightEye.y < y_bordersUp[1]:
            print("Looking to the UPPER screen")
        elif y_bordersDown[0] < centerRightEye.y < y_bordersDown[1]:
            print("Looking to the LOWER screen")
        else:print("Looking to the MIDDLE")
    
    cv2.namedWindow("windowName", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("windowName",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("windowName", img)      
    
    key = cv2.waitKey(1)&0xff
    if key==ord('q'):
        break
    
    
cam.release()
cv2.destroyAllWindows()
