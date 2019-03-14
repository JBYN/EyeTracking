# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:53:24 2019

@author: Jo
"""

import numpy as np

def computeXGradient(img):
    output = []
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]
    
    for y in range(0, rows-1):
        M = img[y]
        output[0] = M[1]-M[0]
        for x in range(1,columns-1):
            output[x] = M[x+1]-M[x-1]
        output[columns-1] = M[columns-1]-M[columns-2]
    return output


