# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""
import cv2
import numpy as np

FAST_EYE_WIDTH = 15

def computeGradient(img):
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]
    gradient = np.zeros((rows, columns,2),np.uint8)
    gradientX = computeXGradient(img)
    gradientY = np.transpose(computeXGradient(np.transpose(img)))
    for y in range(0,rows):
        for x in range(0,columns):
            gradient[y][x] = (gradientX[y][x],gradientY[y][x])
    return gradient

def computeXGradient(img):
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]
    output = np.zeros((rows,columns),np.uint8)
    
    for y in range(0, rows):
        M = img[y]
        output[y][0] = M[1]-M[0]
        for x in range(1,columns-1):
            output[y][x] = M[x+1]-M[x-1]
        output[y][columns-1] = M[columns-1]-M[columns-2]
    
    return output

def gradientMagnitude(gradient):
    rows = np.shape(gradient)[0]
    columns = np.shape(gradient)[1]
    gradientMagnitudes = np.zeros((rows,columns),np.uint8)
    
    for y in range(0, rows):
        for x in range(0, columns):
            gradientMagnitudes[y][x]= np.sqrt(np.square(gradient[y][x][0])+np.square(gradient[y][x][1])) 
    return gradientMagnitudes

def computeDynamicThreshold(data, deviationFactor):
    mean,sigma = cv2.meanStdDev(data)
    stdDev = sigma / np.sqrt(np.size(data))
    return deviationFactor*stdDev + mean
    
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
#    print("blur: " + str(blur))
    weight = 255 - blur #inverted
#    print("weight: " + str(weight))
    return weight

def testPossibleCenters(weight, gradient):
    rows = np.shape(weight)[0]
    columns = np.shape(weight)[1]
    results = np.zeros((rows,columns),np.uint8)
#    print("SHAPE_RESULTS: " + str(np.shape(results)))
    print("PossibleCenters_BEGIN")
    #For each possible centre in the image
    for cy in range(0,rows):
        for cx in range(0,columns):
            d = displacementVector(gradient,(cx,cy))
            magnitudes = gradientMagnitude(d)
            d_norm = normalizeGradient(d,magnitudes,0)
            results[cy][cx] = np.mean(np.square(d_norm*gradient)*weight[cy][cx])
    print("PossibleCenters_END")
#    print("RESULTS: " + str(results))
    return results

#def testPossibleCentersFormula(cx, cy, weight, gradientX, gradientY):
#    output = 0
#    #The sum of the displacement vector * gradient squared for each pixel position
#    for y in range(0, np.shape(weight)[0]):
#        for x in range(0,np.shape(weight)[1]):
#            if x != cx or y != cy or gradientX[y][x] != 0 or gradientY[y][x] != 0:
#                gradient_vector = [gradientX[y][x],gradientY[y][x]]
#                
#                #displacement vector
#                #vector from the possible center to the gradient origin
#                dx = x - cx
#                dy = y - cy
#                #normalize the distance
#                magnitude = round(np.sqrt(np.square(dx) + np.square(dy)))
#                dx_norm = dx/magnitude
#                dy_norm = dy/magnitude
#                displacementVector = [[dx_norm],[dy_norm]]
#                
#                dotProduct = max(0,np.dot(gradient_vector,displacementVector))
#                #Square and multiply by the weight
#                output += np.square(dotProduct)*weight[cy][cx]
#    return output

def displacementVector(gradient, c):
    X = np.zeros((np.shape(gradient)[0],np.shape(gradient)[1],np.shape(gradient)[2]),np.uint8)
    for y in range(0,np.shape(gradient)[0]):
        for x in range(0,np.shape(gradient)[1]):
            X[y][x] = (x,y)
    displacement_vector = X-c
    return displacement_vector

def removeEdges(centers):
#    print("shapeCenters: " + str(np.shape(centers)))
    rows = np.shape(centers)[0]
    columns = np.shape(centers)[1]
    #Remove the horizontal borders
    centers[0] = [0]*columns
    centers[rows-1] = [0]*columns
    #Remove the vertical borders
    centers = np.transpose(centers)
    centers[0] = [0]*rows
    centers[columns-1] = [0]*rows
    
    return np.transpose(centers)

def unscalePoint(point, img_eye):
    ratio = FAST_EYE_WIDTH/(np.shape(img_eye)[1])
    x = round(point[0] / ratio)
    y = round(point[1] / ratio)
    return (x,y)
        