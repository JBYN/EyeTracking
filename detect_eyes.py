#source
#https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
import cv2
#import pickle

#constants to show errors on the image
font = cv2.FONT_ITALIC
color = (0,0,255)
stroke = 2
line_type = cv2.LINE_AA

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")#importing the training data
#
#labels = {"person_name": 1}
#with open("labels.pickle", 'rb') as f:
#    og_labels = pickle.load(f)
#    labels = {v:k for k,v in og_labels.items()}
    
cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop

while True:
    b, img = cam.read()
    #to make it possible to detect faces the capture has to be in grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if faces == ():
        cv2.putText(img,"Not able to detect a face", (0,50),font,1,color,stroke, line_type)
    else:
        for (x,y,w,h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #pixels of the region of interest
            roi_color = img[y:y+h, x:x+w]
            
            #detecting of eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_color_R = (0,0,255) #Red
            eye_color_L = (255,0,0) #Blue
            eye_stroke = 2
            nr_eyes = 0  
            eye_1 = []
            eye_2 = []
            for (eye_x,eye_y,eye_w,eye_h) in eyes:
                    nr_eyes += 1
                    if nr_eyes == 1:
                        eye_1 = [eye_x,eye_y,eye_h, eye_w]
                    elif nr_eyes == 2:
                        eye_2 = [eye_x,eye_y,eye_h, eye_w] 
                        
            if eye_1 != [] and eye_2 != []:            
                if eye_1[0] < eye_2[0] and eye_1[3] < eye_2[0]:
                    #right eye
                    cv2.rectangle(roi_color, (eye_1[0], eye_1[1]), (eye_1[0] + eye_1[3], eye_1[1] + eye_1[2]), eye_color_R, eye_stroke)
                    right_eye = roi_gray[eye_1[1]:eye_1[1]+eye_1[2], eye_1[0]:eye_1[0]+eye_1[3]]
                    #left eye
                    cv2.rectangle(roi_color, (eye_2[0], eye_2[1]), (eye_2[0] + eye_2[3], eye_2[1] + eye_2[2]), eye_color_L, eye_stroke)
                elif eye_2[0] < eye_1[0] and eye_2[3] < eye_1[0]:
                    #right eye
                    cv2.rectangle(roi_color, (eye_2[0], eye_2[1]), (eye_2[0] + eye_2[3], eye_2[1] + eye_2[2]), eye_color_R, eye_stroke)
                    right_eye = roi_gray[eye_2[1]:eye_2[1]+eye_2[2], eye_2[0]:eye_2[0]+eye_2[3]]
                    #left eye
                    cv2.rectangle(roi_color, (eye_1[0], eye_1[1]), (eye_1[0] + eye_1[3], eye_1[1] + eye_1[2]), eye_color_L, eye_stroke)
        #            left_eye = roi_gray[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
                else: cv2.putText(img,"Not able to detect two eyes", (0,50),font,1,color,stroke, line_type)            
            else:
                cv2.putText(img,"Not able to detect two eyes", (0,50),font,1,color,stroke, line_type)
    
    
    #        recognizing
    #        confidence is the distance in the training model. The closer the object is the more sure the program is.
    #        id_, conf = recognizer.predict(roi_gray) 
    #        print("id: " + labels[id_])
    #        print("conf: " + str(conf))
    #        if not conf <= 50:
    #            name = "unknow"
    #        else:
    #        name = labels[id_] + " ---->  " + str(conf)
    #            
    #        font = cv2.FONT_HERSHEY_SIMPLEX
    #        color = (0,255,0)
    #        stroke = 2
    #        cv2.putText(img, name,(x,y-10), font, 1, color, stroke, cv2.LINE_AA)
    #        
    #        starting_coord = (x,y)
    #        ending_coord = (x+w,y+h)
    #        color = (0,255,0) #BGR
    #        stroke = 2
    #        cv2.rectangle(img,starting_coord,ending_coord,color,stroke)
    
    if b:
        cv2.namedWindow("Window", cv2.WND_PROP_ASPECT_RATIO)
        cv2.setWindowProperty("Window", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_NORMAL)
        cv2.imshow("Window",img)
    else:
        print("The camera is not working")
        break
    key = cv2.waitKey(1)&0xff
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
