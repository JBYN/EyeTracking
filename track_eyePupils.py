#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import findEyeCenter as detect
import callibrating

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
x_bordersMiddle = callibrating.get_callibrationResults(callibrating.middle[0])
y_bordersMiddle = callibrating.get_callibrationResults(callibrating.middle[1])
#Left borders
x_bordersLeft = callibrating.get_callibrationResults(callibrating.left[0])
y_bordersLeft = callibrating.get_callibrationResults(callibrating.left[1])
#Right borders
x_bordersRight = callibrating.get_callibrationResults(callibrating.right[0])
y_bordersRight = callibrating.get_callibrationResults(callibrating.right[1])

#screen = np.zeros((800,1600,3),np.uint8)
cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop

while True:
    print("start")
    b,img = cam.read()
    eyes = detect.detect_2eyesOf1person(face_cascade,eye_cascade,b,img)
    
    #detecting eye pupils
    if eyes != None:
        right_eye = eyes[1]
        center = detect.detect_eyeCenter(right_eye)
        if x_bordersMiddle[0] < center[0] < x_bordersMiddle[1] and y_bordersMiddle[0] < center[1] < y_bordersMiddle[1]:
            print("Looking in the middle of the screen")
        elif x_bordersLeft[0] < center[0] < x_bordersLeft[1] and y_bordersLeft[0] < center[1] < y_bordersLeft[1]: 
            print("Looking to the left of the screen")
        elif x_bordersRight[0] < center[0] < x_bordersRight[1] and y_bordersRight[0] < center[1] < y_bordersRight[1]:            
            print("Looking to the right of the screen")
    
    cv2.namedWindow("windowName", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("windowName",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("windowName", img)      
    
    key = cv2.waitKey(1)&0xff
    if key==ord('q'):
        break
    
    
cam.release()
cv2.destroyAllWindows()
