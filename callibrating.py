# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:39:50 2019

@author: Jo
"""

import cv2
import numpy as np
import findEyeCenter as detect
import classes as c
import statistics as stat

CALLIBRATING_DATA = 50

#Lists to collect the callibration data
middle = list()
middle.append(list()) #x values
middle.append(list()) #y values

left = list()
left.append(list()) #x values
left.append(list()) #y values

right = list()
right.append(list()) #x values
right.append(list()) #y values

up = list()
up.append(list()) #x values
up.append(list()) #y values

down = list()
down.append(list()) #x values
down.append(list()) #y values

#Start callibration
def callibrate():
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    
    #selecting the camera
    cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop
    
    while True:
        view,img = cam.read() #open the camera
       
        parameters = c.Eye_Parameters(face_cascade,eye_cascade,view,img)
        if callibrate_dataCollection(parameters): 
            break
 
        cv2.waitKey(1)&0xff

    cam.release()
    cv2.destroyAllWindows()
    print("END of Callibration")

#Adding the different callibration data to the proper lists
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: True: End of the collecting the data; False: Collecting the data is still going on
def callibrate_dataCollection(parameters: c.Eye_Parameters):
    callibrating_background = np.zeros((800,1600,3),np.uint8)+255 #White background
    
    if len(middle[0]) < CALLIBRATING_DATA :
        results_middle = callibrate_middle(callibrating_background,parameters)
        if results_middle != None:
            print("Middle: " + str(len(middle[0])))
            middle[0].append(results_middle.leftPupil.x)
            middle[0].append(results_middle.rightPupil.x)
            middle[1].append(results_middle.leftPupil.y)
            middle[1].append(results_middle.rightPupil.y)
    elif len(left[0]) < CALLIBRATING_DATA:    
        results_left = callibrate_left(callibrating_background,parameters)
        if results_left != None:
            left[0].append(results_left.leftPupil.x)
            left[0].append(results_left.rightPupil.x)
            left[1].append(results_left.leftPupil.y)
            left[1].append(results_left.rightPupil.y)
    elif len(right[0]) < CALLIBRATING_DATA:
        results_right = callibrate_right(callibrating_background,parameters)
        if results_right != None:
            right[0].append(results_right.leftPupil.x)
            right[0].append(results_right.rightPupil.x)
            right[1].append(results_right.leftPupil.y)
            right[1].append(results_right.rightPupil.y)
    elif len(up[0]) < CALLIBRATING_DATA:
        results_up = callibrate_up(callibrating_background,parameters)
        if results_up != None:
            up[0].append(results_up.leftPupil.x)
            up[0].append(results_up.rightPupil.x)
            up[1].append(results_up.leftPupil.y)
            up[1].append(results_up.rightPupil.y)
    elif len(down[0]) < CALLIBRATING_DATA:
        results_down = callibrate_down(callibrating_background,parameters)
        if results_down != None:
            down[0].append(results_down.leftPupil.x)
            down[0].append(results_down.rightPupil.x)
            down[1].append(results_down.leftPupil.y)
            down[1].append(results_down.rightPupil.y)
    else:
        return True
    
    return False

#Callibration for looking to the middle of the screen.
#Getting the center of the two eyes when they are looking to the middle of the screen
#@param callibrating_screen: the background of the screen
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: [ x and y value of the center of the right eye (x,y), x ana y value of the center of the left eye (x,y)]        
def callibrate_middle(callibrating_screen,parameters):
    x = int(callibrating_screen.shape[1]/2)
    y = int(callibrating_screen.shape[0]/2)
    show_callibratingScreen(callibrating_screen,x,y)
    
    return get_eyeCenters(parameters)


#Callibration for looking to the left of the screen.
#Getting the center of the two eyes when they are looking to the left of the screen
#@param callibrating_screen: the background of the screen
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: [ x and y value of the center of the right eye (x,y), x ana y value of the center of the left eye (x,y)]        
def callibrate_left(callibrating_screen, parameters):
    x = 10
    y = int(callibrating_screen.shape[0]/2)
    show_callibratingScreen(callibrating_screen,x,y) 
    
    return get_eyeCenters(parameters)
    

#Callibration for looking to the right of the screen.
#Getting the center of the two eyes when they are looking to the right of the screen
#@param callibrating_screen: the background of the screen
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: [ x and y value of the center of the right eye (x,y), x ana y value of the center of the left eye (x,y)]        
def callibrate_right(callibrating_screen,parameters):
    x = int(callibrating_screen.shape[1])-10
    y = int(callibrating_screen.shape[0]/2)
    show_callibratingScreen(callibrating_screen,x,y) 
    
    return get_eyeCenters(parameters)
    
def callibrate_up(callibrating_screen,parameters):
    x = int(callibrating_screen.shape[1]/2)
    y = 10
    show_callibratingScreen(callibrating_screen,x,y) 
    
    return get_eyeCenters(parameters)

def callibrate_down(callibrating_screen,parameters):
    x = int(callibrating_screen.shape[1]/2)
    y = int(callibrating_screen.shape[0])-10
    show_callibratingScreen(callibrating_screen,x,y) 
    
    return get_eyeCenters(parameters)
   
#Showing a screen with a focus point to get callibrating values
#@param callibrating_screen: the background of the screen
#@param x: the x value of the center of the focus point
#@param y: the y value of the center of the focus point
#@param windowName: the name of the screen
def show_callibratingScreen(callibrating_screen,x,y):
    #focus point
    screen = cv2.rectangle(callibrating_screen,(x-5,y-5),(x+5,y+5),(0,0,0),10)
    
    cv2.namedWindow("callibrate", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("callibrate",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("callibrate", screen) 

#Defining the region that is relevant
#@param data: a list of data
#@return: [low bound, high bound]
def get_callibrationResults(data):
    mu = stat.mu(data)
    sigma = stat.sigma(data,mu)
    return [mu - sigma, mu + sigma]

def get_eyeCenters(parameters):
    pupils = detect.detect_eyeCenter(parameters)
    return pupils
    