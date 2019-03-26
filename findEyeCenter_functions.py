# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""
import cv2
import numpy as np

FAST_EYE_WIDTH = 50

def get_weight(eye,weightBlurSize):
    blur = cv2.GaussianBlur(eye,(weightBlurSize,weightBlurSize),0,0)
#    print("blur: " + str(blur))
    weight = 255 - blur #inverted
#    print("weight: " + str(weight))
    return weight

def unscalePoint(point, img_eye):
    ratio = FAST_EYE_WIDTH/(np.shape(img_eye)[1])
    x = round(point[0] / ratio)
    y = round(point[1] / ratio)
    return (x,y)
        