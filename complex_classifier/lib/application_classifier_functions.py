# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Functions needed to apply the finished classifier to real data

"""
import numpy as np
import os
import torch
import scipy
import sys

from lib.standardization_functions import std_data
from lib.scipy_fit_functions import get_scipy_pred
from lib.pole_objective_functions import complex_conjugate_pole_pair_obj
from lib.architectures import FC1
from lib.pole_classifier import Pole_Classifier
from lib.curve_calc_functions import pole_curve_calc2


def prepare_data9(data_x, data_y, with_bounds=False, do_std=False, std_path=None):
    '''
    Prepare Data (a single sample) for the NN Classifier (9 Classes)
    
    Note: This is used, when the finished classifier is applied to real data. 
    For data generation use lib.training_data_generation_classifier.create_training_data_classifier instead.
    
    data_x: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Gridpoints
    
    data_y: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Function Values
    
    with_bounds: bool, default=False
        Shall Scipy curve_fit be restricted to boundaries given by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max
    
    do_std: bool, default=False
        Shall the data be standardized to the standardization, that was used to train the Classifier?
    
    std_path: str, default=None
        Only needed if do_std=True: Path to the folder containing variances.npy and means.npy (names must be exactly like this!)
        
    returns: ndarray of shape (1, 69+n), where n is the number of gridpoints
        contains: mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c, data_y
    '''
    data_x = data_x.reshape(-1)
    data_y = data_y.reshape(-1)

    try:
        params_1r   = get_scipy_pred(pole_class=0, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_1r = [np.nan, np.nan]
    try:
        params_1c   = get_scipy_pred(pole_class=1, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_1c = [np.nan, np.nan, np.nan, np.nan]
    try:
        params_2r   = get_scipy_pred(pole_class=2, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_2r = [np.nan, np.nan, np.nan, np.nan]
    try:
        params_1r1c = get_scipy_pred(pole_class=3, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_1r1c = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        params_2c   = get_scipy_pred(pole_class=4, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_2c = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        params_3r   = get_scipy_pred(pole_class=5, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_3r = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        params_2r1c = get_scipy_pred(pole_class=6, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_2r1c = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        params_1r2c = get_scipy_pred(pole_class=7, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_1r2c = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        params_3c   = get_scipy_pred(pole_class=8, data_x=data_x, data_y=data_y, with_bounds=with_bounds)
    except:
        print('An error occured!')
        params_3c = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]          
    
    # Calculate predicted curves
    out_re_1r   = pole_curve_calc2(pole_class=0, pole_params=params_1r, data_x=data_x)
    out_re_1c   = pole_curve_calc2(pole_class=1, pole_params=params_1c, data_x=data_x)
    out_re_2r   = pole_curve_calc2(pole_class=2, pole_params=params_2r, data_x=data_x)
    out_re_1r1c = pole_curve_calc2(pole_class=3, pole_params=params_1r1c, data_x=data_x)
    out_re_2c   = pole_curve_calc2(pole_class=4, pole_params=params_2c, data_x=data_x)
    out_re_3r   = pole_curve_calc2(pole_class=5, pole_params=params_3r, data_x=data_x)
    out_re_2r1c = pole_curve_calc2(pole_class=6, pole_params=params_2r1c, data_x=data_x)
    out_re_1r2c = pole_curve_calc2(pole_class=7, pole_params=params_1r2c, data_x=data_x)
    out_re_3c   = pole_curve_calc2(pole_class=8, pole_params=params_3c, data_x=data_x)
    
    # Calculate mse_s
    mse_1r   = np.mean((data_y - out_re_1r)  **2, axis=1)
    mse_1c   = np.mean((data_y - out_re_1c)  **2, axis=1)
    mse_2r   = np.mean((data_y - out_re_2r)  **2, axis=1)
    mse_1r1c = np.mean((data_y - out_re_1r1c)**2, axis=1)
    mse_2c   = np.mean((data_y - out_re_2c)  **2, axis=1)
    mse_3r   = np.mean((data_y - out_re_3r)  **2, axis=1)
    mse_2r1c = np.mean((data_y - out_re_2r1c)**2, axis=1)
    mse_1r2c = np.mean((data_y - out_re_1r2c)**2, axis=1)
    mse_3c   = np.mean((data_y - out_re_3c)  **2, axis=1)
    
    # Apply log10 to mse_s, to get them on a similiar scale, which makes training easier
    mse_1r   = np.log10(mse_1r)
    mse_1c   = np.log10(mse_1c)
    mse_2r   = np.log10(mse_2r)
    mse_1r1c = np.log10(mse_1r1c)
    mse_2c   = np.log10(mse_2c)
    mse_3r   = np.log10(mse_3r)
    mse_2r1c = np.log10(mse_2r1c)
    mse_1r2c = np.log10(mse_1r2c)
    mse_3c   = np.log10(mse_3c)
    
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c, data_y))
    data_x = data_x.reshape(1,-1)
    # Apply standardization
    if do_std:
        data_x = std_data(data=data_x, std_path=std_path, with_mean=True)
        
    return data_x


def get_classifier_preds(data_x, data_y, with_bounds, do_std, model_path):
    '''
    Get predictions from trained Pole Classifier(s) 
    
    data_x: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Gridpoints
    
    data_y: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Function Values
    
    with_bounds: bool
        Shall Scipy curve_fit be restricted to boundaries given by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max? (see also: prepare_data9)
    
    do_std: bool
        Shall the data be standardized to the standardization, that was used to train the Classifier?
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Classifier.
            
            If do_std=True, there must additionally be a subdirectory called 'data' containing the standardization files called "variances.npy" and "means.npy".
    
    returns: int, int, ndarray of shape (l,), where l is the number of checkpoints in the "models" subdirectory
        Class Predictions: Hard Averaging, Soft Averaging, Array with Predictions from the individual Checkpoints
    '''
    # Data Preparation
    data_classifier = prepare_data9(data_y=data_y, data_x=data_x, with_bounds=with_bounds, do_std=do_std, std_path=os.path.join(model_path, 'data/'))
    
    # Get predictions from Classifiers
    class_pred = []
    for filename in os.listdir(os.path.join(model_path, 'models/')):
        model = Pole_Classifier.load_from_checkpoint(os.path.join(model_path, 'models/', filename))
        model.eval()
        pred = model(torch.from_numpy(data_classifier.astype('float32')))
        class_pred.append( pred.detach().numpy() )
        del model
    class_pred = np.array(class_pred)
    class_pred = np.swapaxes(class_pred,0,1)
    
    # Step 3a: Hard Averaging - Take most frequent prediction (=mode)
    class_pred_hard = np.argmax(class_pred, axis=2)
    class_pred_hard
    class_pred_hard = scipy.stats.mode(class_pred_hard, axis=1)[0] 
    class_pred_hard = torch.from_numpy(class_pred_hard)
    
    # Step 3b: Soft Averaging - sum over probabilities and the take argmax
    class_pred_soft = np.sum(class_pred, axis=1)
    class_pred_soft = np.argmax(class_pred_soft, axis=1).reshape((-1,1))
    class_pred_soft = torch.from_numpy(class_pred_soft)
    
    return class_pred_soft.item(), class_pred_hard.item(), np.argmax(class_pred, axis=2).reshape(-1)

















