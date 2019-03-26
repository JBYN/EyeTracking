#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import numpy as np
import findEyeCenter_functions as hf

SMOOTH_FACTOR = 0.005
GRADIENT_THRESHOLD = 0.3
BLUR_WEIGHT_SIZE = 5
POSTPROCESS_THRESHOLD = 0.9

def detect_eyeCenter(parameters):
    output = []
    eyes = detect_2eyesOf1person(parameters)
    
    if eyes != None:
        for i in range(0,np.shape(eyes)[0]):
            eye = cv2.resize(eyes[i][0],(hf.FAST_EYE_WIDTH, int(((hf.FAST_EYE_WIDTH)/np.shape(eyes[i][0])[1])*np.shape(eyes[i][0])[0])))

            #get weight
            weight = hf.get_weight(eye,BLUR_WEIGHT_SIZE)
            _,maxValW,_,maxLocW = cv2.minMaxLoc(weight)
            print("DarkValue: " + str(maxValW))
            xy_originalEye = hf.unscalePoint(maxLocW,eyes[i][0])
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
def detect_2eyesOf1person(parameters):
    two_eyes = False #Not 2 eyes detected yet       
    face_cascade = parameters[0]
    eye_cascade = parameters[1]
    b = parameters[2]
    img = parameters[3]
    
    #to make it possible to detect faces the capture has to be in grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5,0|cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT)
    
    if faces != ():
        print("FACE detected!")
        #getting the coördinates of the detected faces
        for (x,y,w,h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #pixels of the region of interest
            roi_color = img[y:y+h, x:x+w]
            
            #preprocessing
            sigma = SMOOTH_FACTOR * w;
            roi_gray = cv2.GaussianBlur(roi_gray, (0,0), sigma);        
            
            eyes = findEyes(eye_cascade, roi_gray)
            
            if eyes != None:
                print("EYES detected!")
                #Check that the detected eyes are a left and right eye
                checkedEyes = check4LeftAndRightEye(eyes)
                if checkedEyes != None:
                    two_eyes == True
                    img_leftEye = roi_gray[checkedEyes[0][1]:checkedEyes[0][1]+checkedEyes[0][2], checkedEyes[0][0]:checkedEyes[0][0]+checkedEyes[0][3]]
                    img_rightEye = roi_gray[checkedEyes[1][1]:checkedEyes[1][1]+checkedEyes[1][2], checkedEyes[1][0]:checkedEyes[1][0]+checkedEyes[1][3]]
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

    if not b:
        print("The camera is not working")
        
    return None
    
def findEyes(eye_cascade, img):
    #looking for eyes in the region where the faces are detected
    eyes = eye_cascade.detectMultiScale(img)
     
    eye_1 = []
    eye_2 = []
    nr_eyes = 0
    if eyes != ():
        #getting the coördinates of the detected eyes
        for (x,y,w,h) in eyes:
            nr_eyes += 1
            if nr_eyes == 1:
                eye_1 = [x,y,h,w]
            elif nr_eyes == 2:
                eye_2 = [x,y,h,w]
                return [eye_1,eye_2]
    return None

def check4LeftAndRightEye(eyes):
    eye_1 = eyes[0]
    eye_2 = eyes[1]
    
    if eye_1[0] < eye_2[0] and eye_1[3] < eye_2[0]:
        rightEye = eye_1
        leftEye = eye_2
        return [leftEye,rightEye]
    elif eye_2[0] < eye_1[0] and eye_2[3] < eye_1[0]:
        rightEye = eye_2
        leftEye = eye_1
        return [leftEye,rightEye]
    else:
        return None
