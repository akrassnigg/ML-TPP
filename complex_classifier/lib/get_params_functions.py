# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:06:39 2021

@author: siegfriedkaidisch

Functions, that generate pole configurations

"""
import numpy as np
import sys

from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min


def get_params(m, typ: str,
               re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
               coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
               coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min):
    '''
    Generates parameters of real or cc pole pairs. 
    
    WARNING: Real poles are treated as cc pole pairs with zero imaginary parts. 
    Thus, in the end, you have to multiply coeff_re by 2.
    
    m: int>0
        The number of configurations to be created
        
    typ: str: 'r' or 'c'
        Shall the generated poles be real or cc pairs?
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric, defaults read from parameters file
        Define a box. Parameter configurations outside this box are dropped
        
    returns: numpy.ndarray of shape (m, 4)
        The generated pole configurations
    '''
    
    if typ == 'r':
         part_re  = np.random.uniform(re_min, re_max, size=(m))
         
         part_im  = np.zeros((m))
         
         # Get Coeffs from +/- [coeff_min, coeff_max]
         sign = np.random.choice([-1,1], size=(m))
         tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         coeff_re = np.random.uniform(tmp_min, tmp_max, size=(m)) 
         
         coeff_im = np.zeros((m))
         
    elif typ == 'c':
        part_re  = np.random.uniform(re_min, re_max, size=(m))
        
        part_im  = np.random.uniform(im_min, im_max, size=(m))
        
        # Get Coeffs from +/- [coeff_min, coeff_max]
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        coeff_re = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        coeff_im = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        
    else:
        sys.exit("Undefined type.")
        
    return (np.array([part_re, part_im, coeff_re, coeff_im])).transpose()


def get_train_params(pole_class:int, m:int):
    '''
    For a given pole class, return m parameter configurations
    
    pole_class: int: 0-8
        The pole class
        
    m: int
        The number of configurations to be generated
        
    returns: numpy.ndarray of shape (m,k), where k is determined by the pole Class and m is the number of samples
        The generated configurations
    '''
    if pole_class == 0:
        # a single pole on the real axis
        params_1 = get_params(m, 'r')
        params  = params_1
    elif pole_class == 1:
        # a complex conjugated pole pair
        params_1 = get_params(m, 'c')
        params  = params_1
    elif pole_class == 2:
        # two real poles
        params_1 = get_params(m, 'r') 
        params_2 = get_params(m, 'r')
        params  = np.hstack([params_1, params_2])
    elif pole_class == 3:
        # a real pole and a cc pole pair
        params_1 = get_params(m, 'r') 
        params_2 = get_params(m, 'c')
        params  = np.hstack([params_1, params_2])
    elif pole_class == 4:
        # two cc pole pairs
        params_1 = get_params(m, 'c')  
        params_2 = get_params(m, 'c')
        params  = np.hstack([params_1, params_2])
    elif pole_class == 5:
        # three real poles
        params_1 = get_params(m, 'r')
        params_2 = get_params(m, 'r')
        params_3 = get_params(m, 'r')
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 6:
        # two real poles and one cc pole pair
        params_1 = get_params(m, 'r')
        params_2 = get_params(m, 'r')
        params_3 = get_params(m, 'c')
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 7:
        # one real pole and two cc pole pairs
        params_1 = get_params(m, 'r')
        params_2 = get_params(m, 'c')
        params_3 = get_params(m, 'c')
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 8:
        # three cc pole pairs
        params_1 = get_params(m, 'c')
        params_2 = get_params(m, 'c')
        params_3 = get_params(m, 'c')
        params  = np.hstack([params_1, params_2, params_3])
    else:
        # should not happen
        sys.exit("Undefined label.")
    return params



