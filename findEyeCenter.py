#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import numpy as np
import classes as c

SMOOTH_FACTOR = 0.005
GRADIENT_THRESHOLD = 0.3
BLUR_WEIGHT_SIZE = 5
POSTPROCESS_THRESHOLD = 0.9

def detect_eyeCenter(parameters: c.Eye_Parameters):
    output = []
    eyes = detect_2eyesOf1person(parameters)
    
    if eyes != None:
        for i in range(0,np.shape(eyes)[0]):
            eye = eyes[i][0]
            #get position of the pupil by looking for the darkest spot
            pos = get_positionPupil(eye,BLUR_WEIGHT_SIZE)
#            print("DarkValue: " + str(maxValW))
            xy_originalEye = pos
            xy_face = (eyes[i][1][0]+xy_originalEye[0], eyes[i][1][1]+xy_originalEye[1])
            output.append((xy_originalEye,xy_face))
        
        return output
    return None

#Detecting the 2 eyes on an image of a person using haarcascades.
#When the 2 eyes are detected, the region where they are detected are returned.
#Otherwise the value None will be returned
#@param face_cascade: is a CascadeClassifier which is used to detect faces
#@param eye_cascade: is a CascadeClassifier which is used to detect eyes
#@param b:
#@param img: the image where the eyes has to be detect on
#@return: [right eye, left eye]
def detect_2eyesOf1person(parameters: c.Eye_Parameters):
    two_eyes = False #Not 2 eyes detected yet       
    
    #to make it possible to detect faces the capture has to be in grayscale. 
    gray = cv2.cvtColor(parameters.image, cv2.COLOR_BGR2GRAY)
    faces = parameters.face_cascade.detectMultiScale(gray,1.3,5,0|cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT)
    
    if faces != ():
        print("FACE detected!")
        #getting the coördinates of the detected faces
        for (x,y,w,h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #pixels of the region of interest
            roi_color = parameters.image[y:y+h, x:x+w]
            
            #preprocessing
            sigma = SMOOTH_FACTOR * w;
            roi_gray = cv2.GaussianBlur(roi_gray, (0,0), sigma);        
            
            p_eyes = findEyes(parameters.eye_cascade, roi_gray)
            
            if p_eyes != None:
                print("EYES detected!")
                #Check that the detected eyes are a left and right eye
                p_checkedEyes = check4LeftAndRightEye(p_eyes)
                if p_checkedEyes != None:
                    two_eyes == True
                    p_leftEye = p_checkedEyes.leftEye
                    p_rightEye = p_checkedEyes.rightEye
                    img_leftEye = roi_gray[p_leftEye.y:p_leftEye.y + p_leftEye.height, p_leftEye.x:p_leftEye.x + p_leftEye.width]
                    img_rightEye = roi_gray[p_rightEye.y:p_rightEye.y + p_rightEye.height, p_rightEye.x:p_rightEye.x + p_rightEye.width]
                    #Keeping track of the postion of the left corner of the eye on the original image
                    xy_cornerLeftEye = (checkedEyes[0][0]+x,checkedEyes[0][1]+y)
                    xy_cornerRightEye = (checkedEyes[1][0]+x,checkedEyes[1][1]+y)
                    
                    #Constants for indicator
                    eye_color_R = (0,0,255) #Red
                    eye_color_L = (255,0,0) #Blue
                    eye_stroke = 2
                    #indicator
                    ind_leftEye = cv2.rectangle(roi_color, (checkedEyes[0][0], checkedEyes[0][1]), (checkedEyes[0][0] + checkedEyes[0][3], checkedEyes[0][1] + checkedEyes[0][2]), eye_color_L, eye_stroke)
                    ind_rightEye = cv2.rectangle(roi_color, (checkedEyes[1][0], checkedEyes[1][1]), (checkedEyes[1][0] + checkedEyes[1][3], checkedEyes[1][1] + checkedEyes[1][2]), eye_color_R, eye_stroke)
                    return [(img_leftEye,xy_cornerLeftEye),(img_rightEye,xy_cornerRightEye)]
            else:
                print("Not able to detect two eyes")

    else:
        print("No face detected")

    if not parameters.view:
        print("The camera is not working")
        
    return None
    
def findEyes(eye_cascade: cv2.CascadeClassifier, img: np.ndarray) -> c.Position2Eyes:
    #looking for eyes in the region where the faces are detected
    eyes = eye_cascade.detectMultiScale(img)

    nr_eyes = 0
    if eyes != ():
        #getting the coördinates of the detected eyes
        for (x,y,w,h) in eyes:
            nr_eyes += 1
            if nr_eyes == 1:
                p_eye1 = c.Rectangle(x,y,w,h)
            elif nr_eyes == 2:
                p_eye2 = c.Rectangle(x,y,w,h)
                return c.Position2Eyes(p_eye1,p_eye2)
    return None

def check4LeftAndRightEye(eyes: c.Position2Eyes) -> c.Position2Eyes_LR:
    eye_1 = eyes.eye1
    eye_2 = eyes.eye2
    
    if eye_1.x < eye_2.x and (eye_1.x + eye_1.w) < eye_2.x:
        p_rightEye = eye_1
        p_leftEye = eye_2
        return c.Position2Eyes_LR(p_leftEye,p_rightEye)
    elif eye_2.x < eye_1.x and (eye_2.x + eye_2.w) < eye_1.x:
        p_rightEye = eye_2
        p_leftEye = eye_1
        return c.Position2Eyes_LR(p_leftEye,p_rightEye)
    else:
        return None
    
def get_positionPupil(eye,weightBlurSize):
    #pre-processing
    blur = cv2.GaussianBlur(eye,(weightBlurSize,weightBlurSize),0,0)
    #get min
    _,_,minLoc,_ = cv2.minMaxLoc(blur)
    
    return minLoc

