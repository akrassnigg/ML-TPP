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
from lib.scipy_fit_functions import pole_config_organize
from lib.standardization_functions import std_data_new, std_data


def create_training_data_regressor(mode, length, pole_class, data_x, data_dir):
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
        
    data_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored/ loaded from
        
    returns: None or 2 numpy.ndarrays of shapes (length,n) and (length,k), where n is the number of gridpoints and k depends on the pole class
        see: mode
    '''
    # Generate pole configurations
    params = get_train_params(pole_class=pole_class, num=length)
    # Calculate the pole curves
    out_re = pole_curve_calc(pole_class=pole_class, pole_params=params, data_x=data_x)
    # Remove rows, that contain only zeros (imaginary parts of real poles)
    params = params[~np.all(params == 0, axis=1)]
    # Organize pole configurations
    params = pole_config_organize(pole_class=pole_class, pole_params=params)
    # Transpose params
    params = params.transpose()

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
