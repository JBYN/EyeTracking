# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:30:50 2019

@author: Jo
"""
import cv2
import numpy as np

class Eye_Parameters():
    
    def __init__(self,face_cascade: cv2.CascadeClassifier, eye_cascade: cv2.CascadeClassifier, view: bool, image: np.ndarray):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.image = image
        self.view  = view
        
class Rectangle():
    
    def __init__(self, x_left_upperCorner: int, y_left_upperCorner: int, width: int, height: int):
        self.x = x_left_upperCorner
        self.y = y_left_upperCorner
        self.width = width
        self.height = height
        
class Position2Eyes():
    
    def __init__(self, position_eye1: Rectangle, position_eye2: Rectangle):
        self.eye1 = position_eye1
        self.eye2 = position_eye2
        
class Position2Eyes_LR(Position2Eyes):
    
    def __init__(self, position_eyeL: Rectangle, position_eyeR: Rectangle):
        self.leftEye = position_eyeL
        self.rightEye = position_eyeR