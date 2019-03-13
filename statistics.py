# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:50:43 2019

@author: Jo
"""
import math

#determine the mean value of data
#@param data: a list with the data
#@return: type float the mean value
def mu(data):
    return sum(data)/len(data)

#Determine the variance of the data
#@param data: a list with the data
#@param mu: the mean value of the data
#@return: type float
def sigma(data,mu):
    i = 0
    for x in data:
        data[i] = (data[i] - mu)*(data[i] - mu)
        i += 1
        
    var = sum(data)/len(data)
    
    return math.sqrt(var)