#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Select specific data samples for the classifier
Usage:
    1. Generate unconstrained data with create_data_classifier
    2. Apply contraints (e.g. fact_classifier and dst_min_classifier, see parameters file) using this file

"""
import os
import numpy as np
from pathlib import Path
import shutil

from parameters import data_dir_classifier, standard_re
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import fact_classifier, dst_min_classifier
from lib.training_data_generation_classifier import drop_small_poles_2, drop_near_poles
from lib.standardization_functions import rm_std_data, std_data_new

def drop_outside_box(pole_class, pole_params):
    '''
    Drops parameter configurations, whose parameters are outside the box specified by re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (k,m), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept      
    '''
    
    if pole_class==0:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)

    if pole_class==1:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (im_min       < pole_params[1,:])          * (pole_params[1,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[3,:]))  * (np.abs(pole_params[3,:])  < coeff_im_max)

    if pole_class==2:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)

    if pole_class==3:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (im_min       < pole_params[5,:])          * (pole_params[5,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[7,:]))  * (np.abs(pole_params[7,:])  < coeff_im_max)

    if pole_class==4:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (im_min       < pole_params[1,:])          * (pole_params[1,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[3,:]))  * (np.abs(pole_params[3,:])  < coeff_im_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (im_min       < pole_params[5,:])          * (pole_params[5,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[7,:]))  * (np.abs(pole_params[7,:])  < coeff_im_max)

    if pole_class==5:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[8,:])          * (pole_params[8,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[10,:])) * (np.abs(pole_params[10,:]) < coeff_re_max)
        
    if pole_class==6:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[8,:])          * (pole_params[8,:]  < re_max)
        keep_indices *= (im_min       < pole_params[9,:])          * (pole_params[9,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[10,:])) * (np.abs(pole_params[10,:]) < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[11,:])) * (np.abs(pole_params[11,:]) < coeff_im_max)
        
    if pole_class==7:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (im_min       < pole_params[5,:])          * (pole_params[5,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[7,:]))  * (np.abs(pole_params[7,:])  < coeff_im_max)
        keep_indices *= (re_min       < pole_params[8,:])          * (pole_params[8,:]  < re_max)
        keep_indices *= (im_min       < pole_params[9,:])          * (pole_params[9,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[10,:])) * (np.abs(pole_params[10,:]) < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[11,:])) * (np.abs(pole_params[11,:]) < coeff_im_max)
        
    if pole_class==8:
        keep_indices  = (re_min       < pole_params[0,:])          * (pole_params[0,:]  < re_max)
        keep_indices *= (im_min       < pole_params[1,:])          * (pole_params[1,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[2,:]))  * (np.abs(pole_params[2,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[3,:]))  * (np.abs(pole_params[3,:])  < coeff_im_max)
        keep_indices *= (re_min       < pole_params[4,:])          * (pole_params[4,:]  < re_max)
        keep_indices *= (im_min       < pole_params[5,:])          * (pole_params[5,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[6,:]))  * (np.abs(pole_params[6,:])  < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[7,:]))  * (np.abs(pole_params[7,:])  < coeff_im_max)
        keep_indices *= (re_min       < pole_params[8,:])          * (pole_params[8,:]  < re_max)
        keep_indices *= (im_min       < pole_params[9,:])          * (pole_params[9,:]  < im_max)
        keep_indices *= (coeff_re_min < np.abs(pole_params[10,:])) * (np.abs(pole_params[10,:]) < coeff_re_max)
        keep_indices *= (coeff_im_min < np.abs(pole_params[11,:])) * (np.abs(pole_params[11,:]) < coeff_im_max)
        
    return keep_indices


if __name__ == '__main__':
    # Make a backup of the data
    src_path = Path(data_dir_classifier)
    trg_path = Path(data_dir_classifier + 'backup/')
    trg_path.mkdir(exist_ok=True, parents=True)
    for each_file in src_path.glob('*.*'): # grabs all files
        shutil.copy(each_file, trg_path)
        
    # Import the data, generated with create_data_classifier
    data_x = np.load(os.path.join(data_dir_classifier, "various_poles_data_classifier_x.npy"), allow_pickle=True).astype('float32')
    labels = np.load(os.path.join(data_dir_classifier, "various_poles_data_classifier_y.npy"), allow_pickle=True).astype('int64').reshape((-1,1))
    params = np.load(os.path.join(data_dir_classifier, "various_poles_data_classifier_params.npy"), allow_pickle=True).astype('float32')
    
    # Remove standardization from data_x
    data_x = rm_std_data(data=data_x, with_mean=True, std_path=data_dir_classifier, name_var="variances.npy", name_mean="means.npy")
    
    # Apply restrictions
    new_data_x = []
    new_labels = []
    new_params = []
    for pole_class in range(9):
        indices    = (labels==pole_class).reshape(-1)
        labels_tmp = labels[indices]
        data_x_tmp = data_x[indices]
        params_tmp = params[indices]
        
        # transpose
        params_tmp_2 = params_tmp.copy()
        params_tmp_2 = np.atleast_2d(params_tmp_2)
        params_tmp_2 = params_tmp_2.transpose()
        
        keep_indices  = drop_small_poles_2(pole_class=pole_class, pole_params=params_tmp_2, data_x=standard_re, fact=fact_classifier)
        keep_indices *= drop_near_poles(pole_class=pole_class, pole_params=params_tmp_2, dst_min=dst_min_classifier)
        keep_indices *= drop_outside_box(pole_class=pole_class, pole_params=params_tmp_2)
        
        new_data_x.append(data_x_tmp[keep_indices])
        new_labels.append(labels_tmp[keep_indices])
        new_params.append(params_tmp[keep_indices])
        
    new_data_x = np.vstack(new_data_x)
    new_labels = np.vstack(new_labels)
    new_params = np.vstack(new_params)
    
    # Standardize Inputs
    new_data_x = std_data_new(new_data_x, with_mean=True, std_path=data_dir_classifier)
    
    # Save training data
    np.save(os.path.join(data_dir_classifier, 'various_poles_data_classifier_x.npy'), new_data_x)
    np.save(os.path.join(data_dir_classifier, 'various_poles_data_classifier_y.npy'), new_labels)
    np.save(os.path.join(data_dir_classifier, 'various_poles_data_classifier_params.npy'), new_params)
    print("Successfully saved x data of shape ", np.shape(new_data_x))
    print("Successfully saved y data of shape ", np.shape(new_labels))
    print("Successfully saved params data of shape ", np.shape(new_params))
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    