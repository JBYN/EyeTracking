#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
import numpy as np
import classes as c

SMOOTH_FACTOR = 0.005
GRADIENT_THRESHOLD = 0.3
BLUR_WEIGHT_SIZE = 5
POSTPROCESS_THRESHOLD = 0.9

def detect_eyeCenter(parameters: c.Eye_Parameters) -> c.EyePupils:
    face = detect_2eyesOf1person(parameters)
    
    if face != None:
        cv2.imshow("leftEye",face.leftEye.imgEye)
        #get position of the pupil by looking for the largest dark area
        posPupils = get_positionPupils(face,BLUR_WEIGHT_SIZE)
        
        return posPupils
    return None

#Detecting the 2 eyes on an image of a person using haarcascades.
#When the 2 eyes are detected, the region where they are detected are returned.
#Otherwise the value None will be returned
#@param face_cascade: is a CascadeClassifier which is used to detect faces
#@param eye_cascade: is a CascadeClassifier which is used to detect eyes
#@param b:
#@param img: the image where the eyes has to be detect on
#@return: [right eye, left eye]
def detect_2eyesOf1person(parameters: c.Eye_Parameters) -> c.Face:
    two_eyes = False #Not 2 eyes detected yet       
    
    #to make it possible to detect faces the capture has to be in grayscale. 
    gray = cv2.cvtColor(parameters.image, cv2.COLOR_BGR2GRAY)
    faces = parameters.face_cascade.detectMultiScale(gray,1.3,5,0|cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT)
    
    if faces != ():
        print("FACE detected!")
        #getting the coördinates of the detected faces
        for (x,y,w,h) in faces:
            p_face = c.Rectangle(x,y,w,h)
            roi_gray = gray[p_face.y:p_face.y + p_face.height, p_face.x:p_face.x + p_face.width] #pixels of the region of interest
            roi_color = parameters.image[p_face.y:p_face.y + p_face.height, p_face.x:p_face.x + p_face.width]
            
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
                    #Get the postions of the two eyes so they can be isolated
                    p_leftEye = p_checkedEyes.leftEye
                    p_rightEye = p_checkedEyes.rightEye
                    img_leftEye = roi_gray[p_leftEye.y:p_leftEye.y + p_leftEye.height, p_leftEye.x:p_leftEye.x + p_leftEye.width]
                    img_rightEye = roi_gray[p_rightEye.y:p_rightEye.y + p_rightEye.height, p_rightEye.x:p_rightEye.x + p_rightEye.width]
                  
                    leftEye = c.Eye(p_leftEye,img_leftEye)
                    rightEye = c.Eye(p_rightEye, img_rightEye)
                    
                    face = c.Face(p_face,roi_color,leftEye,rightEye)
                    
                    #Constants for indicator
                    eye_color_R = (0,0,255) #Red
                    eye_color_L = (255,0,0) #Blue
                    eye_stroke = 2
                    #indicator
                    ind_leftEye = cv2.rectangle(roi_color, (p_leftEye.x, p_leftEye.y), (p_leftEye.x + p_leftEye.width, p_leftEye.y + p_leftEye.width), eye_color_L, eye_stroke)
                    ind_rightEye = cv2.rectangle(roi_color, (p_rightEye.x, p_rightEye.y), (p_rightEye.x + p_rightEye.width, p_rightEye.y + p_rightEye.width), eye_color_R, eye_stroke)
                    return face
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
            elif (nr_eyes == 2) and np.abs(p_eye1.height-y)<= 50:
                p_eye2 = c.Rectangle(x,y,w,h)
                return c.Position2Eyes(p_eye1,p_eye2)
            else: nr_eyes -=1
    return None

def check4LeftAndRightEye(eyes: c.Position2Eyes) -> c.Position2Eyes_LR:
    eye_1 = eyes.eye1
    eye_2 = eyes.eye2

    if eye_1.x < eye_2.x and (eye_1.x + eye_1.width) < eye_2.x:
        p_rightEye = eye_1
        p_leftEye = eye_2
        return c.Position2Eyes_LR(p_leftEye,p_rightEye)
    elif eye_2.x < eye_1.x and (eye_2.x + eye_2.width) < eye_1.x:
        p_rightEye = eye_2
        p_leftEye = eye_1
        return c.Position2Eyes_LR(p_leftEye,p_rightEye)
    return None
    
def get_positionPupils(face: c.Face, weightBlurSize: int) -> c.EyePupils:
    p_leftPupil = get_positionPupil(face.leftEye.imgEye,weightBlurSize)
    p_rightPupil = get_positionPupil(face.rightEye.imgEye, weightBlurSize)
#    #pre-processing
#    blur_leftEye = cv2.GaussianBlur(face.leftEye.imgEye ,(weightBlurSize,weightBlurSize),0,0)
#    blur_rightEye = cv2.GaussianBlur(face.rightEye.imgEye, (weightBlurSize,weightBlurSize), 0,0)
#    #Remove edges
#    leftEye = removeEdges(blur_leftEye)
#    rightEye = removeEdges(blur_rightEye)
#    #get min 
#    _,_,minLoc_L,_ = cv2.minMaxLoc(leftEye)
#    _,_,minLoc_R,_ = cv2.minMaxLoc(rightEye)
    if p_leftPupil != None and p_rightPupil != None:
        print("EYECENTERS detected!")
        posPupils = scalePosition(face,p_leftPupil,p_rightPupil)
        return posPupils
    return None

def get_positionPupil(eye: np.ndarray, weightBlurSize: int)->c.Point:
    #pre-processing
    blur_Eye = cv2.GaussianBlur(eye ,(weightBlurSize,weightBlurSize),0,0)
    _,threshold = cv2.threshold(blur_Eye, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("THRESHOLD", threshold)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #get position pupil
    if contours !=None:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            return c.Point(int(x + w/2), int(y + h/2))
    return None

def scalePosition(face: c.Face, pos_left: c.Point, pos_right: c.Point) -> c.EyePupils:
    leftPupil_scaled = c.Point((pos_left.x + face.leftEye.pEye.x + face.pFace.x),(pos_left.y + face.leftEye.pEye.y + face.pFace.y))
    rightPupil_scaled = c.Point((pos_right.x + face.rightEye.pEye.x + face.pFace.x),(pos_right.y + face.rightEye.pEye.y + face.pFace.y))
    
    return c.EyePupils(leftPupil_scaled,rightPupil_scaled)

def removeEdges(eye: np.ndarray)->np.ndarray:
    eye = removeHorizontalEdges(eye)
    eye = np.transpose(removeHorizontalEdges(np.transpose(eye)))
    return eye

def removeHorizontalEdges(eye: np.ndarray)->np.ndarray:
    eye[0] = [255]*np.shape(eye)[0]
    eye[np.shape(eye)[0]-1] = [255]*np.shape(eye)[0]
    return eye