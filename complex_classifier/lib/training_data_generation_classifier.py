#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Generate data for the classifier

"""
import numpy as np
import os

from lib.get_params_functions import get_train_params
from lib.scipy_fit_functions import get_all_scipy_preds_dataprep
from lib.curve_calc_functions import pole_curve_calc, pole_curve_calc2
from lib.standardization_functions import std_data_new
from lib.diverse_functions import mse, drop_not_finite_rows
from parameters import fact_classifier


def drop_poles_fact(pole_class, pole_params, point_x, fact):
    '''
    Drops parameter configureations, that contain poles, whose out_re_i is a factor fact smaller, than out_re_i of the other poles in the sample
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (k,m), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    point_x: float
        Point, where the different out_re_i are compared
        
    fact: numeric>=1
        The factor to compare out_re_i from the different poles per sample
        
    returns: ndarray of shape (k,m*), where m* is the number of left samples
        Parameters specifying the Pole Configuration and satisfying min(out_re_i)*fact >= max(out_re_i)      
    '''
    data_x = np.array([point_x])
    
    if pole_class == 0 or pole_class == 1:
        new_pole_params = pole_params
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        params1 = pole_params[0:4,:]
        params2 = pole_params[4:8,:]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, data_x=data_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, data_x=data_x)
        data_y_1 = np.abs(data_y_1)
        data_y_2 = np.abs(data_y_2)
        data_y   = np.concatenate((data_y_1, data_y_2), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
        new_pole_params = pole_params[:,keep_indices]
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        params1 = pole_params[0:4,:]
        params2 = pole_params[4:8,:]
        params3 = pole_params[8:12,:]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, data_x=data_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, data_x=data_x)
        data_y_3 = pole_curve_calc(pole_class=1, pole_params=params3, data_x=data_x)
        data_y_1 = np.abs(data_y_1)
        data_y_2 = np.abs(data_y_2)
        data_y_3 = np.abs(data_y_3)
        data_y   = np.concatenate((data_y_1, data_y_2, data_y_3), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
        new_pole_params = pole_params[:,keep_indices]
    return new_pole_params

def drop_poles_fact2(pole_class, pole_params, data_x, fact):
    '''
    Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
    
    Note: The difference to drop_poles_fact is, that here data_x is a vector, opposed to point_x in drop_poles_fact.
    Out_re is then calculated as the sum over all points in data_x.
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (k,m), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    data_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    fact: numeric>=1
        The factor to compare out_re_i from the different poles per sample
        
    returns: ndarray of shape (k,m*), where m* is the number of left samples
        Parameters specifying the Pole Configuration and satisfying min(out_re_i)*fact >= max(out_re_i)      
    '''
    data_x = data_x.reshape(-1)
    
    if pole_class == 0 or pole_class == 1:
        new_pole_params = pole_params
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        params1 = pole_params[0:4,:]
        params2 = pole_params[4:8,:]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, data_x=data_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, data_x=data_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        data_y   = np.concatenate((data_y_1, data_y_2), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
        new_pole_params = pole_params[:,keep_indices]
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        params1 = pole_params[0:4,:]
        params2 = pole_params[4:8,:]
        params3 = pole_params[8:12,:]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, data_x=data_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, data_x=data_x)
        data_y_3 = pole_curve_calc(pole_class=1, pole_params=params3, data_x=data_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        data_y_3 = np.sum(np.abs(data_y_3), axis=1).reshape(-1,1)
        data_y   = np.concatenate((data_y_1, data_y_2, data_y_3), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
        new_pole_params = pole_params[:,keep_indices]
    return new_pole_params

def create_training_data_classifier(length, data_x, with_bounds, data_dir):
    '''
    Creates training data for the NN classifier and saves it to the disk
    
    length: int>0
        The number of samples to be generated
        
    data_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    with_bounds: bool, default=False
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_min, coeff_max, re_min, re_max, im_min, im_max?
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored
        
    returns: None
    '''
    # List of the pole classes
    pole_classes = [0,1,2,3,4,5,6,7,8]
    
    # calculate the number of samples per class and the rest
    num  = int(np.floor(length/len(pole_classes)))
    rest = int(length - num * len(pole_classes)) 
    
    # out_re will contain the real part of the pole curves; labels is the labels (0,1,2,..8)
    out_re = []
    labels = []
    for pole_class in pole_classes:
        # Get pole configurations of the current pole class and append them to out_re and labels
        params = get_train_params(pole_class=pole_class, num=num)
        params = drop_poles_fact2(pole_class=pole_class, pole_params=params, data_x=data_x, fact=fact_classifier)
        
        out_re.append(pole_curve_calc(pole_class=pole_class, pole_params=params, data_x=data_x))
        labels.append(np.ones(num)*pole_class)
    # Convert lists to numpy arrays
    out_re = np.vstack(out_re)
    labels = np.hstack(labels)
    
    # Get Scipy predictions for each sample
    print('Getting SciPy predictions...')
    out_re, labels, params_1r, params_1c, params_2r, params_1r1c, params_2c, \
    params_3r, params_2r1c, params_1r2c, params_3c = get_all_scipy_preds_dataprep(data_x, out_re, labels, with_bounds=with_bounds)
    
    # Calculate out_re for the different predicted pole configurations
    out_re_1r   = pole_curve_calc2(pole_class=0, pole_params=params_1r,   data_x=data_x)
    out_re_1c   = pole_curve_calc2(pole_class=1, pole_params=params_1c,   data_x=data_x)
    out_re_2r   = pole_curve_calc2(pole_class=2, pole_params=params_2r,   data_x=data_x)
    out_re_1r1c = pole_curve_calc2(pole_class=3, pole_params=params_1r1c, data_x=data_x)
    out_re_2c   = pole_curve_calc2(pole_class=4, pole_params=params_2c,   data_x=data_x)
    out_re_3r   = pole_curve_calc2(pole_class=5, pole_params=params_3r,   data_x=data_x)
    out_re_2r1c = pole_curve_calc2(pole_class=6, pole_params=params_2r1c, data_x=data_x)
    out_re_1r2c = pole_curve_calc2(pole_class=7, pole_params=params_1r2c, data_x=data_x)
    out_re_3c   = pole_curve_calc2(pole_class=8, pole_params=params_3c,   data_x=data_x)
    
    # Calculate the different MSEs
    mse_1r   = mse(out_re, out_re_1r,   ax=1).reshape(-1,1)
    mse_1c   = mse(out_re, out_re_1c,   ax=1).reshape(-1,1)
    mse_2r   = mse(out_re, out_re_2r,   ax=1).reshape(-1,1)
    mse_1r1c = mse(out_re, out_re_1r1c, ax=1).reshape(-1,1)
    mse_2c   = mse(out_re, out_re_2c,   ax=1).reshape(-1,1)
    mse_3r   = mse(out_re, out_re_3r,   ax=1).reshape(-1,1)
    mse_2r1c = mse(out_re, out_re_2r1c, ax=1).reshape(-1,1)
    mse_1r2c = mse(out_re, out_re_1r2c, ax=1).reshape(-1,1)
    mse_3c   = mse(out_re, out_re_3c,   ax=1).reshape(-1,1)
    
    # Apply log10 to the MSEs to bring them to a similiar scale
    mse_1r   = np.log10(mse_1r)
    mse_1c   = np.log10(mse_1c)
    mse_2r   = np.log10(mse_2r)
    mse_1r1c = np.log10(mse_1r1c)
    mse_2c   = np.log10(mse_2c)
    mse_3r   = np.log10(mse_3r)
    mse_2r1c = np.log10(mse_2r1c)
    mse_1r2c = np.log10(mse_1r2c)
    mse_3c   = np.log10(mse_3c)
    
    # Transpose the parameter arrays for the next steps
    params_1r   = params_1r.transpose()
    params_1c   = params_1c.transpose()
    params_2r   = params_2r.transpose()
    params_1r1c = params_1r1c.transpose()
    params_2c   = params_2c.transpose()
    params_3r   = params_3r.transpose()
    params_2r1c = params_2r1c.transpose()
    params_1r2c = params_1r2c.transpose()
    params_3c   = params_3c.transpose()
    
    # Get rid of possible infinities that can occurr after log10 for very small MSE below machine accuracy
    mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, \
    params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c, \
    out_re, labels = drop_not_finite_rows(
        mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, 
        params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c, 
        out_re, labels
        )
    
    # Everything together the gives data_x that is used to train the classifier
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c, out_re))
    
    # Reshape labels array
    labels = labels.reshape(-1)

    # Standardize Inputs
    data_x = std_data_new(data_x, with_mean=True, std_path=data_dir)
    
    # Save training data
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_x.npy'), data_x)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_y.npy'), labels)
    print("Successfully saved x data of shape ", np.shape(data_x))
    print("Successfully saved y data of shape ", np.shape(labels))

    return




#
