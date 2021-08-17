# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 10:57:17 2021

@author: siegfriedkaidisch

A bunch of functions to be used for the pole classifier and regressor

"""
import numpy as np


def mse(arr1, arr2, ax):
    '''
    Calculate the MSE between two arrays along a given axis
    
    arr1, arr2: numpy.ndarrays of the same shape
    
    ax: int
        The axis along which the MSE shall be computed
    
    returns: numpy.ndarray with shape of arr1, reduced by axis=ax
        The MSEs
    '''
    return np.mean((arr1 - arr2)**2,axis=ax)


def drop_not_finite_rows(*arrs):
    '''
    Checks for rows containing "not finite" elements and drops them from the given arrays
    
    *arrs: 2D numpy.ndarrays with the same number of rows
    
    returns: list of 2D numpy.ndarrays with the same number of rows
        The reduced arrays
    '''
    arrs = list(arrs)
    
    arr  = np.concatenate(arrs, axis=1)
    finite = np.isfinite(arr).all(axis=1)
    
    for i in range(len(arrs)):
        arrs[i] = arrs[i][finite]
        
    return arrs