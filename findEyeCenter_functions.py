# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""
import cv2
import numpy as np


def computeXGradient(img):
    output = []
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]
    
    for y in range(0, rows-1):
        M = img[y]
        output.append([])
        output[y].append(M[1]-M[0])
        for x in range(1,columns-1):
            output[y].append(M[x+1]-M[x-1])
        output[y].append(M[columns-1]-M[columns-2])
    return output

def matrixMagnitude(matrix1, matrix2):
    magnitudeMatrix = []
    
    for y in range(0, np.shape(matrix1)[0]-1):
        magnitudeMatrix.append([])
        if np.shape(np.shape(matrix1))[0] > 1:
            for x in range(0, np.shape(matrix1)[1]-1):
                magnitudeMatrix[y].append(magnitude(matrix1[y][x],matrix2[y][x]))
        else:
            magnitudeMatrix[y].append(magnitude(matrix1[y],matrix2[y]))
    return magnitudeMatrix

def magnitude(value1, value2):
    return np.sqrt(np.square(value1)+np.square(value2))

def computeDynamicThreshold(data, deviationFactor):
    mean = np.mean(data)
    sigma = np.std(data)
    stdDev = sigma / np.sqrt(np.shape(data)[0]*np.shape(data)[1])
    return deviationFactor*stdDev + mean
    
