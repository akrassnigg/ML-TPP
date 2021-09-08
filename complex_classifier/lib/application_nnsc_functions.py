# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:21:50 2021

@author: siegfriedkaidisch

Functions to get predictions from NN regressor checkpoints in combination with SciPy's curve_fit

"""
import numpy as np
import os

from lib.application_regressor_functions import get_regressor_pred
from lib.scipy_fit_functions import get_scipy_pred
from parameters import regressor_subdirs


def get_nnsc_pred(pole_class, grid_x, data_y, with_bounds, model_path):
    '''
    Find optimal parameters for fitting data to a given Pole Configuration; 
    First get predictions from a NN Regressor, then give them as p0 to scipy curve_fit

    pole_class: int = 0-8 
        The Class of the Pole Configuration    

    grid_x, data_y: ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    with_bounds: bool
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Regressor.
            
            There must additionally be a subdirectory called 'data' containing the standardization files called 
            "variances.npy" for the inputs and "variances_params.npy", "means_params.npy" for the outputs.
    
    returns: ndarray of shape (k,)
        Optimized parameters of the chosen Pole Configuration/Class
    '''
    # Get NN regressor_pred
    pred_nn   = get_regressor_pred(data_y=data_y, model_path=model_path)[0]

    # Scipy Fit
    pred_nnsc = get_scipy_pred(pole_class=pole_class, grid_x=grid_x, data_y=data_y, with_bounds=with_bounds, p0=pred_nn)

    return pred_nnsc


def get_all_nnsc_preds(grid_x, data_y, with_bounds, model_path):
    '''
    Find optimal parameters for fitting data to all 9 different Pole Configurations; 
    First get predictions from a NN Regressor, then give them as p0 to scipy curve_fit
    
    grid_x, data_y: ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    with_bounds: bool
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    model_path: str
        Path to folder containing all necessary files:
            Must contain a subdirectory for each of the 9 Pole Configurations: 
                '0-1r', '1-1c', '2-2r', '3-1r1c', '4-2c', '5-3r', '6-2r1c', '7-1r2c', '8-3c'
            
            Inside each of these subdirectories:
                There must be a subsubdirectory called 'models' containing atleast one PL checkpoint of the Regressor.
                
                There must additionally be a subsubdirectory called 'data' containing the standardization files called 
                "variances.npy" for the inputs and "variances_params.npy", "means_params.npy" for the outputs.

    returns: list of 9 ndarrays of shapes (k_i,), i=0...8
        Optimized parameters of the different Pole Configurations/Classes
    '''
    preds = []
    for subdir in regressor_subdirs:
        try:
            pred = get_nnsc_pred(model_path=os.path.join(model_path, subdir + '/'), pole_class=int(subdir[0]), 
                                 grid_x=grid_x, data_y=data_y, with_bounds=with_bounds)
        except:
            pred = np.array([])
        preds.append(pred)
    return preds






