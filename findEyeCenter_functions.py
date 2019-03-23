# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""
import cv2
import numpy as np

FAST_EYE_WIDTH = 50

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
    #For each possible centre in the image
    for cy in range(0,rows-1):
        for cx in range(0,columns-1):
            results[cy,cx] = testPossibleCentersFormula(cx,cy,weight, gradientX, gradientY)
    print("PossibleCenters_END")
    return results

def testPossibleCentersFormula(cx, cy, weight, gradientX, gradientY):
    output = 0
    #The sum of the displacement vector * gradient squared for each pixel position
    for y in range(0, np.shape(weight)[0]):
        for x in range(0,np.shape(weight)[1]):
            if x != cx or y != cy or gradientX[y][x] != 0 or gradientY[y][x] != 0:
                gradient_vector = [gradientX[y][x],gradientY[y][x]]
                
                #displacement vector
                #vector from the possible center to the gradient origin
                dx = x - cx
                dy = y - cy
                #normalize the distance
                magnitude = round(np.sqrt(np.square(dx) + np.square(dy)))
                dx_norm = dx/magnitude
                dy_norm = dy/magnitude
                displacementVector = [[dx_norm],[dy_norm]]
                
                dotProduct = max(0,np.dot(gradient_vector,displacementVector))
                #Square and multiply by the weight
                output += int(np.square(dotProduct))*weight[cy][cx]
    return output


#def floodRemoveEdges(centers):
#    rows = np.shape(centers)[0]
#    columns = np.shape(centers)[1]
#    rectangle(centers, cv2.Rect(0,0, rows,columns),255)
#    mask = np.zeros((rows,columns), np.uint8, 255)
#    return None

def unscalePoint(point, img_eye):
    ratio = FAST_EYE_WIDTH/(np.shape(img_eye)[1])
    x = round(point[0] / ratio)
    y = round(point[1] / ratio)
    return (x,y)
        