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
    output = np.zeros((rows,columns,3),np.uint8)
    
    for y in range(0, rows):
        M = img[y]
        output[y] = M[1]-M[0]
        for x in range(1,columns-1):
            output[y] = M[x+1]-M[x-1]
        output[y] = M[columns-1]-M[columns-2]
    return output

def matrixMagnitude(matrix1, matrix2):
    rows = np.shape(matrix1)[0]
    columns = np.shape(matrix1)[1]
    magnitudeMatrix = np.zeros((rows,columns,3),np.uint8)
    
    for y in range(0, rows):
        for x in range(0, columns):
            magnitudeMatrix[y]=magnitude(matrix1[y][x],matrix2[y][x])
    return magnitudeMatrix

def magnitude(value1, value2):
    return np.sqrt(np.square(value1)+np.square(value2))

def computeDynamicThreshold(data, deviationFactor):
    mean,sigma = cv2.meanStdDev(data)
    stdDev = sigma[0] / np.sqrt(np.shape(data)[0]*np.shape(data)[1])
    return deviationFactor*stdDev + mean[0]
    
def normalizeMatrix(matrix,magnitudes,threshold):
    for y in range(0,np.shape(matrix)[0]):
        for x in range(0,np.shape(matrix)[1]):
            gX = matrix[y][x]
            magnitude = magnitudes[y][x][0]
            if magnitude > threshold:
                matrix[y][x] = gX/magnitude
            else:
                matrix[y][x] = 0
    return matrix