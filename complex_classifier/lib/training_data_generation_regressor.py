#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code: Generate data for the regressor

"""
import numpy as np
import sys
import os

from lib.get_params_functions import get_train_params
from lib.curve_calc_functions import pole_curve_calc
from lib.standardization_functions import std_data_new, std_data
from lib.training_data_generation_classifier import drop_small_poles_2, drop_near_poles
from lib.pole_config_organize import pole_config_organize_abs as pole_config_organize


def create_training_data_regressor(mode, length, pole_class, grid_x, data_dir, fact, dst_min,
                                   re_max, re_min, im_max, im_min, 
                                   coeff_re_max, coeff_re_min, 
                                   coeff_im_max, coeff_im_min):
    '''
    Creates training data for the NN regressor
    
    mode: str: 'preparation' or 'update'
        Two different operating types:
            
        Preparation: New data is generated. It is standardized (a new standardization is created) and saved to the disk. This function returns None.
    
        Update: New data is generated and standardized using means and variances files from data_dir. It is then returned by this function.
    
    length: int>0
        The number of samples to be generated
        
    pole_class: int: 0-8
        The pole class
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored/ loaded from
        
    fact: numeric>=1
        Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
        
    dst_min: numeric>=0
        Drops parameter configureations, that contain poles, whose positions are nearer to each other than dst_min (complex, euclidean norm)
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    returns: None or 2 numpy.ndarrays of shapes (length,n) and (length,k), where n is the number of gridpoints and k depends on the pole class
        see: mode
    '''
    # Generate pole configurations
    params = get_train_params(pole_class=pole_class, m=length, 
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
    params = params[drop_small_poles_2(pole_class=pole_class, pole_params=params, grid_x=grid_x, fact=fact)]
    params = params[drop_near_poles(pole_class=pole_class, pole_params=params, dst_min=dst_min)]
    # Calculate the pole curves
    out_re = pole_curve_calc(pole_class=pole_class, pole_params=params, grid_x=grid_x)
    # Organize pole configurations
    params = pole_config_organize(pole_class=pole_class, pole_params=params)

    if mode == 'preparation':
        # Create initial data for training      
        # Standardize Inputs
        out_re = std_data_new(data=out_re, with_mean=False, name_var="variances.npy", std_path=data_dir) 
        #Standardize Outputs
        params = std_data_new(data=params, with_mean=True, name_var="variances_params.npy", 
                              name_mean="means_params.npy", std_path=data_dir)
        # Save training data
        np.save(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), out_re)
        np.save(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), params)
        print("Successfully saved x data of shape ", np.shape(out_re))
        print("Successfully saved y data of shape ", np.shape(params))
        return None
    
    elif mode == 'update':
        # Update dataset during training of the regressor 
        # Standardize Inputs
        out_re = std_data(data=out_re, with_mean=False, name_var="variances.npy", std_path=data_dir)   
        #Standardize Outputs
        params = std_data(data=params, with_mean=True, name_var="variances_params.npy", 
                              name_mean="means_params.npy", std_path=data_dir)
        return out_re, params
    
    else:
        sys.exit("Undefined mode.")





#
