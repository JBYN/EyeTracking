# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""
import cv2
import numpy as np


def computeXGradient(img):
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]
    output = np.zeros((rows,columns),np.uint8)
    
    for y in range(0, rows-1):
        M = img[y]
        output[y][0] = M[1]-M[0]
        for x in range(1,columns-2):
            output[y][x] = M[x+1]-M[x-1]
        output[y][columns-1] = M[columns-1]-M[columns-2]
    
    return output

def gradientMagnitude(gradientX, gradientY):
    rows = np.shape(gradientX)[0]
    columns = np.shape(gradientY)[1]
    gradientMagnitudes = np.zeros((rows,columns),np.uint8)
    
    for y in range(0, rows-1):
        for x in range(0, columns-1):
            gradientMagnitudes[y][x]=magnitude(gradientX[y][x],gradientY[y][x])
    return gradientMagnitudes

def magnitude(value1, value2):
    return np.sqrt(np.square(value1)+np.square(value2))

def computeDynamicThreshold(data, deviationFactor):
    mean,sigma = cv2.meanStdDev(data)
    stdDev = sigma[0] / np.sqrt(np.shape(data)[0]*np.shape(data)[1])
    return deviationFactor*stdDev + mean[0]
    
def normalizeGradient(gradient,magnitudes,threshold):
    for y in range(0,np.shape(gradient)[0]):
        for x in range(0,np.shape(gradient)[1]):
            g = gradient[y][x]
            magnitude = magnitudes[y][x]
            #considering only gradient vectors with a signifacnt magnitude
            if magnitude > threshold:
                gradient[y][x] = g/magnitude
            else:
                gradient[y][x] = 0
    return gradient

def get_weight(eye,weightBlurSize):
    blur = cv2.GaussianBlur(eye,(weightBlurSize,weightBlurSize),0,0)
    print("blur: " + str(blur))
    weight = 255 - blur #inverted
    print("weight: " + str(weight))
    return weight

def testPossibleCenters(weight, gradientX, gradientY):
    rows = np.shape(weight)[0]
    columns = np.shape(weight)[1]
    results = np.zeros((rows,columns),np.uint8)
    print("PossibleCenters_BEGIN")
    for y in range(0,rows-1):
        for x in range(0,columns-1):
            if gradientX[y][x] != 0 or gradientY[y][x] != 0:
                results = testPossibleCentersFormula(x,y,weight, [[gradientX[y][x], gradientY[y][x]]], results)
    print("PossibleCenters_END")
    return results

def testPossibleCentersFormula(x, y, weight, gradient_vector, output):
    for cy in range(0, np.shape(output)[0]):
        for cx in range(0,np.shape(output)[1]):
            if x != cx or y != cy:
                
                #displacement vector
                #vector from the possible center to the gradient origin
                dx = x - cx
                dy = y -cy
                #normalize the distance
                magnitude = np.sqrt(np.square(dx) + np.square(dy))
                dx_norm = dx/magnitude
                dy_norm = dy/magnitude
                displacementVector = [[dx_norm],[dy_norm]]
                
                dotProduct = max(0,np.dot(gradient_vector,displacementVector))
                #Square and multiply by the weight
                output[cy][cx] += int(np.square(dotProduct))*weight[cy][cx]
    return output
