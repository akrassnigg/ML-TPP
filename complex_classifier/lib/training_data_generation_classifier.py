#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Generate data for the classifier

"""
import numpy as np
import os
from pathlib import Path
import shutil

from lib.get_params_functions import get_train_params
from lib.scipy_fit_functions import get_all_scipy_preds_dataprep
from lib.curve_calc_functions import pole_curve_calc, pole_curve_calc2
from lib.diverse_functions import mse, drop_not_finite_rows
from lib.standardization_functions import rm_std_data, std_data_new


def drop_small_poles(pole_class, pole_params, point_x, fact):
    '''
    Drops parameter configureations, that contain poles, whose out_re_i is a factor fact smaller, than out_re_i of the other poles in the sample
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (m,k), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    point_x: float
        Point, where the different out_re_i are compared
        
    fact: numeric>=1
        The factor to compare out_re_i from the different poles per sample
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept        
    '''
    grid_x = np.array([point_x])
    
    if pole_class == 0 or pole_class == 1:
        keep_indices = np.ones(np.shape(pole_params)[0])==1
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_1 = np.abs(data_y_1)
        data_y_2 = np.abs(data_y_2)
        data_y   = np.concatenate((data_y_1, data_y_2), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        params3 = pole_params[:,8:12]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_3 = pole_curve_calc(pole_class=1, pole_params=params3, grid_x=grid_x)
        data_y_1 = np.abs(data_y_1)
        data_y_2 = np.abs(data_y_2)
        data_y_3 = np.abs(data_y_3)
        data_y   = np.concatenate((data_y_1, data_y_2, data_y_3), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
    return keep_indices

def drop_small_poles_2(pole_class, pole_params, grid_x, fact):
    '''
    Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
    
    Note: The difference to drop_poles_fact is, that here grid_x is a vector, opposed to point_x in drop_poles_fact.
    Out_re is then calculated as the sum over all points in grid_x.
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (m,k), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    fact: numeric>=1
        The factor to compare out_re_i from the different poles per sample
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept       
    '''
    grid_x = grid_x.reshape(-1)
    
    if pole_class == 0 or pole_class == 1:
        keep_indices = np.ones(np.shape(pole_params)[0])==1
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        data_y   = np.concatenate((data_y_1, data_y_2), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        params3 = pole_params[:,8:12]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_3 = pole_curve_calc(pole_class=1, pole_params=params3, grid_x=grid_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        data_y_3 = np.sum(np.abs(data_y_3), axis=1).reshape(-1,1)
        data_y   = np.concatenate((data_y_1, data_y_2, data_y_3), axis=1)
        keep_indices = np.max(data_y, axis=1) <= np.min(data_y, axis=1) * fact
    return keep_indices


def drop_small_poles_abs(pole_class, pole_params, grid_x, cutoff):
    '''
    Drops parameter configureations, that contain poles, whose sum(abs(out_re)) is smaller than a cutoff
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (m,k), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    cutoff: numeric>=0
        The cutoff to compare to sum(abs(out_re_i)) from the different poles per sample
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept       
    '''
    grid_x = grid_x.reshape(-1)
    
    if pole_class == 0 or pole_class == 1:
        params1 = pole_params[:,0:4]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        keep_indices = cutoff < data_y_1
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        keep_indices  = cutoff < data_y_1
        keep_indices *= cutoff < data_y_2
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        params1 = pole_params[:,0:4]
        params2 = pole_params[:,4:8]
        params3 = pole_params[:,8:12]
        
        data_y_1 = pole_curve_calc(pole_class=1, pole_params=params1, grid_x=grid_x)
        data_y_2 = pole_curve_calc(pole_class=1, pole_params=params2, grid_x=grid_x)
        data_y_3 = pole_curve_calc(pole_class=1, pole_params=params3, grid_x=grid_x)
        data_y_1 = np.sum(np.abs(data_y_1), axis=1).reshape(-1,1)
        data_y_2 = np.sum(np.abs(data_y_2), axis=1).reshape(-1,1)
        data_y_3 = np.sum(np.abs(data_y_3), axis=1).reshape(-1,1)
        keep_indices  = cutoff < data_y_1
        keep_indices *= cutoff < data_y_2
        keep_indices *= cutoff < data_y_3
    return keep_indices


def drop_near_poles(pole_class, pole_params, dst_min):
    '''
    Drops parameter configureations, that contain poles, whose positions are nearer to each other than dst_min (complex, euclidean norm)
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (m,k), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    dst_min: numeric>=0
        The minimal allowed distance between (complex) pole positions
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept      
    '''
    
    if pole_class == 0 or pole_class == 1:
        keep_indices = np.ones(np.shape(pole_params)[0])==1
    elif pole_class == 2 or pole_class == 3 or pole_class == 4:
        dst_arr = np.sqrt( (pole_params[:,0] - pole_params[:,4])**2 + (pole_params[:,1] - pole_params[:,5])**2 )
        keep_indices = dst_arr > dst_min
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class==8:
        dst_arr = np.sqrt( (pole_params[:,0] - pole_params[:,4])**2 + (pole_params[:,1] - pole_params[:,5])**2 )
        keep_indices = dst_arr > dst_min
        dst_arr = np.sqrt( (pole_params[:,0] - pole_params[:,8])**2 + (pole_params[:,1] - pole_params[:,9])**2 )
        keep_indices *= dst_arr > dst_min
        dst_arr = np.sqrt( (pole_params[:,4] - pole_params[:,8])**2 + (pole_params[:,5] - pole_params[:,9])**2 )
        keep_indices *= dst_arr > dst_min
    return keep_indices


def drop_outside_box(pole_class, pole_params,
                    re_max, re_min, im_max, im_min, 
                    coeff_re_max, coeff_re_min, 
                    coeff_im_max, coeff_im_min):
    '''
    Drops parameter configurations, whose parameters are outside the box specified by re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
    
    pole_class: int = 0-8
        The Class of the Pole Configuration
    
    pole_params: ndarray of shape (m,k), where m is the number of samples
        Parameters specifying the Pole Configuration
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations outside this box are dropped
        
    returns: ndarray of shape (m,)
        Specifies, whether each sample shall be dropped or kept      
    '''
    pole_params = pole_params.transpose()
    
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


def drop_classifier_samples_afterwards(with_mean, data_dir, grid_x,
                                       re_max, re_min, im_max, im_min, 
                                       coeff_re_max, coeff_re_min, 
                                       coeff_im_max, coeff_im_min,
                                       fact=np.inf, dst_min=0.0):
    '''
    Drop unwanted samples by applying drop_small_poles_2, drop_near_poles and drop_outside_box to already
    existing classifier samples, which were previously created using create_training_data_classifier and write the remaining
    samples to the disk.
        
    with_mean: bool
        Shall the mean of each feature be shifted?
        
    data_dir: str
        Directory, where the data shall be loaded from and written to
        
        Must contain files various_poles_data_classifier_x.npy, various_poles_data_classifier_y.npy, 
        various_poles_data_classifier_params.npy, variances.npy and, if with_mean=True, means.npy.
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations outside this box are dropped in drop_outside_box
        
    fact: numeric>=1, default np.inf
        The factor to compare out_re_i from the different poles per sample in drop_small_poles_2
        
    dst_min: numeric>=0, default 0.0
        The minimal allowed distance between (complex) pole positions in drop_near_poles
        
    returns: None     
    '''
    # Make a backup of the data
    src_path = Path(data_dir)
    trg_path = Path(data_dir + 'backup/')
    trg_path.mkdir(exist_ok=True, parents=True)
    for each_file in src_path.glob('*.*'): # grabs all files
        shutil.copy(each_file, trg_path)
    print('Successfully backed up old data')
        
    # Import the data, generated with create_data_classifier
    data_x = np.load(os.path.join(data_dir, "various_poles_data_classifier_x.npy"), allow_pickle=True).astype('float32')
    labels = np.load(os.path.join(data_dir, "various_poles_data_classifier_y.npy"), allow_pickle=True).astype('int64').reshape((-1,1))
    params = np.load(os.path.join(data_dir, "various_poles_data_classifier_params.npy"), allow_pickle=True).astype('float32')
    print("Successfully loaded x data of shape ", np.shape(data_x))
    print("Successfully loaded y data of shape ", np.shape(labels))
    print("Successfully loaded params data of shape ", np.shape(params))

    # Remove standardization from data_x
    data_x = rm_std_data(data=data_x, with_mean=with_mean, std_path=data_dir, name_var="variances.npy", name_mean="means.npy")
    
    new_data_x = []
    new_labels = []
    new_params = []
    for pole_class in range(9):
        indices    = (labels==pole_class).reshape(-1)
        labels_tmp = labels[indices]
        data_x_tmp = data_x[indices]
        params_tmp = params[indices]
        
        # Copy
        params_tmp_2 = params_tmp.copy()
        params_tmp_2 = np.atleast_2d(params_tmp_2)
        
        # Apply restrictions
        keep_indices  = drop_small_poles_2(pole_class=pole_class, pole_params=params_tmp_2, grid_x=grid_x, fact=fact)
        keep_indices *= drop_near_poles(pole_class=pole_class, pole_params=params_tmp_2, dst_min=dst_min)
        keep_indices *= drop_outside_box(pole_class=pole_class, pole_params=params_tmp_2, 
                                         re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                         coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                         coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        
        new_data_x.append(data_x_tmp[keep_indices])
        new_labels.append(labels_tmp[keep_indices])
        new_params.append(params_tmp[keep_indices])
        
    new_data_x = np.vstack(new_data_x)
    new_labels = np.vstack(new_labels)
    new_params = np.vstack(new_params)
    
    # Standardize Inputs
    new_data_x = std_data_new(new_data_x, with_mean=with_mean, std_path=data_dir)
    
    # Save training data
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_x.npy'), new_data_x)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_y.npy'), new_labels)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_params.npy'), new_params)
    print("Successfully saved x data of shape ", np.shape(new_data_x))
    print("Successfully saved y data of shape ", np.shape(new_labels))
    print("Successfully saved params data of shape ", np.shape(new_params))
    
    return None

def get_data_x(data_y, grid_x, 
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min,
               with_bounds=True,
               p0='default', method='trf', maxfev=100000, num_tries=1, xtol = 1e-8):
    '''
    Generates the unstandardized network input from the pole curves (out_re)
    
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values at the gridpoints
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    with_bounds: bool, default=True
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    p0: 'default' or 'random', default='default'
        Initial guesses for parameter search. 
        
        If 'default', the SciPy curve_fit default behaviour is used 
        
        If 'random', random guesses are used (use this if num_tries>1)
        
    method: str = 'trf', 'dogbox' or 'lm', default='trf'
        The optimization method
        
    maxfev: int > 0 , default=100000
        Maximal number of function evaluations (see SciPy's curve_fit)
        
    num_tries: int > 0, default=1
        The number of times the fit shall be tried (with varying initial guesses)
        
    xtol: float or list of floats, default 1e-8
        Convergence criterion (see SciPy's curve_fit)    
    
    returns: two numpy.ndarrays of shapes (m,9+60) and (m,len(grid_x))
        data_x (network input) and the function values, not yet normalized
    '''
    data_y = np.atleast_2d(data_y)
    
    # Get Scipy predictions for each sample
    print('Getting SciPy predictions...')
    params_1r, params_1c, params_2r, params_1r1c, params_2c, \
    params_3r, params_2r1c, params_1r2c, params_3c = get_all_scipy_preds_dataprep(grid_x=grid_x, data_y=data_y, 
                                                                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                                                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                                                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                                                                  with_bounds=with_bounds, p0=p0,
                                                                                  method=method, maxfev=maxfev, num_tries=num_tries, 
                                                                                  xtol=xtol)

    # Calculate out_re for the different predicted pole configurations
    out_re_1r   = pole_curve_calc2(pole_class=0, pole_params=params_1r,   grid_x=grid_x)
    out_re_1c   = pole_curve_calc2(pole_class=1, pole_params=params_1c,   grid_x=grid_x)
    out_re_2r   = pole_curve_calc2(pole_class=2, pole_params=params_2r,   grid_x=grid_x)
    out_re_1r1c = pole_curve_calc2(pole_class=3, pole_params=params_1r1c, grid_x=grid_x)
    out_re_2c   = pole_curve_calc2(pole_class=4, pole_params=params_2c,   grid_x=grid_x)
    out_re_3r   = pole_curve_calc2(pole_class=5, pole_params=params_3r,   grid_x=grid_x)
    out_re_2r1c = pole_curve_calc2(pole_class=6, pole_params=params_2r1c, grid_x=grid_x)
    out_re_1r2c = pole_curve_calc2(pole_class=7, pole_params=params_1r2c, grid_x=grid_x)
    out_re_3c   = pole_curve_calc2(pole_class=8, pole_params=params_3c,   grid_x=grid_x)
    
    # Calculate the different MSEs
    mse_1r   = mse(data_y, out_re_1r,   ax=1).reshape(-1,1)
    mse_1c   = mse(data_y, out_re_1c,   ax=1).reshape(-1,1)
    mse_2r   = mse(data_y, out_re_2r,   ax=1).reshape(-1,1)
    mse_1r1c = mse(data_y, out_re_1r1c, ax=1).reshape(-1,1)
    mse_2c   = mse(data_y, out_re_2c,   ax=1).reshape(-1,1)
    mse_3r   = mse(data_y, out_re_3r,   ax=1).reshape(-1,1)
    mse_2r1c = mse(data_y, out_re_2r1c, ax=1).reshape(-1,1)
    mse_1r2c = mse(data_y, out_re_1r2c, ax=1).reshape(-1,1)
    mse_3c   = mse(data_y, out_re_3c,   ax=1).reshape(-1,1)
    
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
    
    # Everything together then gives data_x that is used to train the classifier
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c))
    
    return data_x, data_y

def create_training_data_classifier(length, grid_x, 
                                    re_max, re_min, im_max, im_min, 
                                    coeff_re_max, coeff_re_min, 
                                    coeff_im_max, coeff_im_min,
                                    with_bounds, data_dir, fact, dst_min, 
                                    p0, method, maxfev, num_tries, xtol):
    '''
    Creates training data for the NN classifier and saves it to the disk.
    
    Data is created by performing SciPy fits with different methods, calculating MSEs and concatenating everything
    
    length: int>0 or a list of ints>=0
        The number of samples to be generated (per class if length is a list)
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    with_bounds: list of bools
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored
        
    fact: numeric>=1
        Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
        
    dst_min: numeric>=0
        Drops parameter configureations, that contain poles, whose positions are nearer to each other than dst_min (complex, euclidean norm)
       
    p0: list of strings: 'default' or 'random'
        Initial guesses for parameter search. 
        
        If 'default', the SciPy curve_fit default behaviour is used 
        
        If 'random', random guesses are used (use this if num_tries>1)
        
    method: list of strings: 'trf', 'dogbox' or 'lm'
        The optimization methods (multiple possible, this is, why this is a list)
        
    maxfev: list of ints > 0 
        Maximal number of function evaluations (see SciPy's curve_fit)
        
    num_tries: list of ints > 0
        The number of times the fit shall be tried (with varying initial guesses) 
       
    xtol: list of floats or list of lists of floats
        Convergence criterion (see SciPy's curve_fit)                         
    
    returns: None
    '''
    # List of the pole classes
    pole_classes = [0,1,2,3,4,5,6,7,8]
    
    if isinstance(length, list):
        nums = length
    else:
        # calculate the number of samples per class
        num  = int(np.floor(length/len(pole_classes)))
        nums = [num for i in range(len(pole_classes))]
    
    # out_re will contain the real part of the pole curves; labels is the labels (0,1,2,..8)
    out_re = []
    labels_and_params = []
    for pole_class in pole_classes:
        # Get pole configurations of the current pole class and append them to out_re and labels
        params = get_train_params(pole_class=pole_class, m=nums[pole_class],
                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params = params[drop_small_poles_2(pole_class=pole_class, pole_params=params, grid_x=grid_x, fact=fact)]
        params = params[drop_near_poles(pole_class=pole_class, pole_params=params, dst_min=dst_min)]
        out_re.append(pole_curve_calc(pole_class=pole_class, pole_params=params, grid_x=grid_x))
        
        # also add the exact parameters to the label, because we may use them later (e.g. for selecting certain samples)
        labels_and_params.append(np.hstack([np.ones([params.shape[0], 1])*pole_class, params]))
        
    # Convert lists to numpy arrays
    out_re = np.vstack(out_re)
    # add padding to the labels/params arrays
    max_coll = max( [arr.shape[1] for arr in labels_and_params] )
    for i in range(len(labels_and_params)):
        fillup = int(max_coll - labels_and_params[i].shape[1])
        labels_and_params[i] = np.hstack([labels_and_params[i], np.zeros([labels_and_params[i].shape[0], fillup])])
    labels_and_params = np.vstack(labels_and_params)
    print('Maximum number of samples to be created: ', len(labels_and_params))
    
    ###########################################################################
    # Get data_x
    for i in range(len(method)): #use different SciPy fitting methods
        data_x_i, out_re = get_data_x(data_y=out_re, grid_x=grid_x, 
                            re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                            coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                            coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                            with_bounds=with_bounds[i], 
                            p0=p0[i], method=method[i], maxfev=maxfev[i], num_tries=num_tries[i], xtol=xtol[i])  
        if i==0:
            data_x = data_x_i
        else:
            data_x = np.hstack([data_x, data_x_i])
        
        ## Get rid of possible infinities that can occurr after log10 for very small MSE below machine accuracy and if the sample couldn't be fitted
        data_x, out_re, labels_and_params = drop_not_finite_rows(
                                    data_x, out_re, labels_and_params)
        
    data_x = np.hstack([data_x, out_re])
    ###########################################################################
    
    # Seperate labels and params array
    params = labels_and_params[:,1:]
    labels = labels_and_params[:,0].reshape(-1)

    # Standardize Inputs
    data_x = std_data_new(data_x, with_mean=True, std_path=data_dir)
    
    # Save training data
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_x.npy'), data_x)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_y.npy'), labels)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_params.npy'), params)
    print("Successfully saved x data of shape ", np.shape(data_x))
    print("Successfully saved y data of shape ", np.shape(labels))
    print("Successfully saved params data of shape ", np.shape(params))
    print('Samples per Class:')
    for i in range(9):
        print(str(i) + ': ', np.sum(labels==i))
    return






#
