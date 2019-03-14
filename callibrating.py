# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:39:50 2019

@author: Jo
"""

import cv2
import numpy as np
import findEyeCenter as detect
import statistics as stat

CALLIBRATING_DATA = 10

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

#Start callibration
def callibrate():
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    
    #selecting the camera
    cam = cv2.VideoCapture(0) #0 used as parameter to use the webcam of the computer/laptop
    
    while True:
        b,img = cam.read() #open the camera
        #eyes = detect.detect_2eyesOf1person(face_cascade,eye_cascade,b,img) #detecting eyes
        parameters = [face_cascade,eye_cascade,b,img]
        if callibrate_dataCollection(parameters): 
            break
 
        cv2.waitKey(1)&0xff

    cam.release()
    cv2.destroyAllWindows()
    print("END of Callibration")

#Adding the different callibration data to the proper lists
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: True: End of the collecting the data; False: Collecting the data is still going on
def callibrate_dataCollection(parameters):
    callibrating_background = np.zeros((800,1600,3),np.uint8)+255 #White background
    
    if len(middle[0]) < CALLIBRATING_DATA :
        results_middle = callibrate_middle(callibrating_background,parameters)
        if results_middle != None:
            if results_middle[0] != None:
                print("Middle: " + str(len(middle[0])))
                middle[0].append(results_middle[0][0])
                middle[1].append(results_middle[0][1])
            if results_middle[1] != None:
                middle[0].append(results_middle[1][0])
                middle[1].append(results_middle[1][1])
    elif len(left[0]) < CALLIBRATING_DATA:
        cv2.destroyWindow("Middle")    
        results_left = callibrate_left(callibrating_background,parameters)
        if results_left != None:
            if results_left[0] != None:
                print("Left: " + str(len(left[0])))
                left[0].append(results_left[0][0])
                left[1].append(results_left[0][1])
            if results_left[1] != None:
                left[0].append(results_left[1][0])
                left[1].append(results_left[1][1])
    elif len(right[0]) < CALLIBRATING_DATA:
        cv2.destroyWindow("Left")
        results_right = callibrate_right(callibrating_background,parameters)
        if results_right != None:
            if results_right[0] != None:
                print("Right: " + str(len(right[0])))
                right[0].append(results_right[0][0])
                right[1].append(results_right[0][1])
            if results_right[1] != None:
                right[0].append(results_right[1][0])
                right[1].append(results_right[1][1])
    else:
        cv2.destroyWindow("Right")
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
    show_callibratingScreen(callibrating_screen,x,y,"Middle")
    
    #detecting eye pupils
#    if eyes != None:
    return get_eyeCenters(parameters)
#    else: return None

#Callibration for looking to the left of the screen.
#Getting the center of the two eyes when they are looking to the left of the screen
#@param callibrating_screen: the background of the screen
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: [ x and y value of the center of the right eye (x,y), x ana y value of the center of the left eye (x,y)]        
def callibrate_left(callibrating_screen, parameters):
    x = 10
    y = int(callibrating_screen.shape[0]/2)
    show_callibratingScreen(callibrating_screen,x,y, "Left") 
    
    #detecting eye pupils
    #if eyes != None:
    return get_eyeCenters(parameters)
    #else: return None

#Callibration for looking to the right of the screen.
#Getting the center of the two eyes when they are looking to the right of the screen
#@param callibrating_screen: the background of the screen
#@param eyes: the regions where the eyes can be found. [region of the right eye, region of the left eye]
#@return: [ x and y value of the center of the right eye (x,y), x ana y value of the center of the left eye (x,y)]        
def callibrate_right(callibrating_screen,parameters):
    x = int(callibrating_screen.shape[1])-10
    y = int(callibrating_screen.shape[0]/2)
    show_callibratingScreen(callibrating_screen,x,y,"Right") 
    
    #detecting eye pupils
    #if eyes != None:
    return get_eyeCenters(parameters)
    #else: return None
    
#Showing a screen with a focus point to get callibrating values
#@param callibrating_screen: the background of the screen
#@param x: the x value of the center of the focus point
#@param y: the y value of the center of the focus point
#@param windowName: the name of the screen
def show_callibratingScreen(callibrating_screen,x,y,windowName):
    #focus point
    screen = cv2.rectangle(callibrating_screen,(x-5,y-5),(x+5,y+5),(0,0,0),10)
    
    cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(windowName, screen) 

#Defining the region that is relevant
#@param data: a list of data
#@return: [low bound, high bound]
def get_callibrationResults(data):
    mu = stat.mu(data)
    sigma = stat.sigma(data,mu)
    return [mu - sigma, mu + sigma]

def get_eyeCenters(parameters):
    return detect.detect_eyeCenter(parameters)
    
callibrate()