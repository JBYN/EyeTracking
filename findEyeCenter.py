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
    eyes = detect_2eyesOf1person(parameters)
    
    if eyes != None:
        for i in range(0,np.shape(eyes)[0]):
            gradientX = hf.computeXGradient(eyes[i])
            gradientY = np.transpose(hf.computeXGradient(np.transpose(eyes[i])))
            print("SHAPE_X: " + str(np.shape(gradientX)))
            print("SHAPE_Y: " + str(np.shape(gradientY)))
            magnitudes = hf.gradientMagnitude(gradientX,gradientY)
            threshold = hf.computeDynamicThreshold(magnitudes, GRADIENT_THRESHOLD)
            
            #Normalize gradientX and gradientY
            gradientX_norm = hf.normalizeGradient(gradientX,magnitudes,threshold)
            gradientY_norm = hf.normalizeGradient(gradientY,magnitudes,threshold)
            
            #get weight
            weight = hf.get_weight(eyes[i],BLUR_WEIGHT_SIZE)
            
            #test the possible centers
            numberGradients = np.square(np.shape(gradientX_norm)[0])
            centers = ((hf.testPossibleCenters(weight, gradientX_norm, gradientY_norm))/numberGradients)
            _,maxVal,_,maxLoc = cv2.minMaxLoc(centers)
            
            #postprocessing
            #applying a threshold
            floodThresh = maxVal*POSTPROCESS_THRESHOLD
            floodClone = cv2.threshold(centers,floodThresh,0.0,cv2.THRESH_TOZERO)
            #remove values connected to the borders
            mask = hf.floodRemoveEdges(floodClone)
            
            _,maxVal,_,maxLoc = cv2.minMaxLoc(centers,mask)
            
            cv2.end.WORKS_FINE
            
#    return unscalePoint(maxLoc,eyes[])
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
                    
                    #Constants for indicator
                    eye_color_R = (0,0,255) #Red
                    eye_color_L = (255,0,0) #Blue
                    eye_stroke = 2
                    #indicator
                    ind_leftEye = cv2.rectangle(roi_color, (checkedEyes[0][0], checkedEyes[0][1]), (checkedEyes[0][0] + checkedEyes[0][3], checkedEyes[0][1] + checkedEyes[0][2]), eye_color_L, eye_stroke)
                    ind_rightEye = cv2.rectangle(roi_color, (checkedEyes[1][0], checkedEyes[1][1]), (checkedEyes[1][0] + checkedEyes[1][3], checkedEyes[1][1] + checkedEyes[1][2]), eye_color_R, eye_stroke)
                    return [img_leftEye,img_rightEye] 
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





###############################################################################
#Old method to detect the eyecenter
###############################################################################

##Detect the circle in the image. This circle represents the pupil of the eye.
##The x and y values of the center of the circle and the processed image are returned.
##When no circle is detected the value None will be returned.
##@param eye determine the circles on this image. (Image of an eye)
##@return [x,y,image]
#def detect_eyeCenter(eye):
#    #preprocessing the images
#    print("preprocessing")
#    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY,0)
#    res_eye = cv2.resize(eye_gray,(400,400), interpolation = cv2.INTER_CUBIC)
#    blur_eye = cv2.GaussianBlur(res_eye,(3,3),0)
#    crop_eye = blur_eye[100:300,0:400]
#    
#    #change the contrast
##    test = np.zeros(crop_eye.shape, crop_eye.dtype)
##    alpha = 2.5;
##    beta = 0;
##    for y in range(crop_eye.shape[0]):
##        for x in range(crop_eye.shape[1]):
##            test[y,x] = np.clip(alpha*crop_eye[y,x] + beta, 0, 255)
##    
##    print("ArrayTest: " + str(test))
#    print("passed preprocessing")
#    cv2.imwrite("blur_eye20.png",crop_eye)
#    
#    #looking for circles with Hough transform
#    print("detect pupil")
#    eye_pupils = cv2.HoughCircles(crop_eye,cv2.HOUGH_GRADIENT,1,15,50,15,30,60)
#    if np.all(eye_pupils) != None:  
#        if str(eye_pupils) != str([[50][0][0][0]]):
#            print("ERROR 50 opgevangen")
#        else:
#            eye_pupils = np.uint16(np.around(eye_pupils))
#            print("origineel: " + str(eye_pupils))
#            print("hhh: " + str(eye_pupils[0,:])) 
#            for pupil in eye_pupils[0,:]:
#                print("pupil: " + str(pupil))
#                print("x: " + str(pupil[0]))
#                print("y: " + str(pupil[1]))
#                print("radius: " + str(pupil[2]))
#                
#                cv2.circle(crop_eye,(pupil[0],pupil[1]),int(pupil[2]/3),(0,255,0))
#                cv2.imshow("New",crop_eye)
#                
#                return [pupil[0],pupil[1],crop_eye]