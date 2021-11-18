# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:21:50 2021

@author: siegfriedkaidisch

Functions to get predictions from NN regressor checkpoints in combination with SciPy's curve_fit

"""
import numpy as np
import os, sys

from lib.application_regressor_functions import get_regressor_pred
from lib.scipy_fit_functions import get_scipy_pred_dual
from lib.pole_config_organize import pole_config_organize_abs_dual
from lib.pole_config_organize import add_zero_imag_parts_dual

def check_inside_bounds(pole_class, pole_params, 
                        re_max, re_min, im_max, im_min, 
                        coeff_re_max, coeff_re_min, 
                        coeff_im_max, coeff_im_min):
    '''
    Checks, whether all values in pole_params are within their given bounds
    
    pole_class: int = 0-8 
        The Class of the Pole Configuration  
        
    pole_params: numpy.ndarray of shape (k,), where k depends on the pole_class (e.g k=4 for pole_class=0)
        Parameters specifying the pole configuration
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Boundaries
        
    returns: Bool
    '''
    max_params = np.array([re_max,im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max])
    min_params = np.array([re_min,im_min,-coeff_re_max,-coeff_im_max,-coeff_re_max,-coeff_im_max])
    if pole_class in [0,1]:
        min_params = np.tile(min_params, (1))
        max_params = np.tile(max_params, (1))
    elif pole_class in [2,3,4]:
        min_params = np.tile(min_params, (2))
        max_params = np.tile(max_params, (2))
    elif pole_class in [5,6,7,8]:
        min_params = np.tile(min_params, (3))
        max_params = np.tile(max_params, (3))
    else:
        sys.exit("Undefined label.")    
        
    # remove imag parts of real poles    
    if   pole_class == 0:
        max_params = max_params[[0,2,4]]
        min_params = min_params[[0,2,4]]
    elif pole_class == 2:
        max_params = max_params[[0,2,4, 6,8,10]]
        min_params = min_params[[0,2,4, 6,8,10]]
    elif pole_class == 3:
        max_params = max_params[[0,2,4, 6,7,8,9,10,11]]
        min_params = min_params[[0,2,4, 6,7,8,9,10,11]]
    elif pole_class == 5:
        max_params = max_params[[0,2,4, 6,8,10, 12,14,16]]
        min_params = min_params[[0,2,4, 6,8,10, 12,14,16]]
    elif pole_class == 6:
        max_params = max_params[[0,2,4, 6,8,10, 12,13,14,15,16,17]]
        min_params = min_params[[0,2,4, 6,8,10, 12,13,14,15,16,17]]
    elif pole_class == 7:
        max_params = max_params[[0,2,4, 6,7,8,9,10,11, 12,13,14,15,16,17]]
        min_params = min_params[[0,2,4, 6,7,8,9,10,11, 12,13,14,15,16,17]]
    
    inside_bounds = np.all(min_params <= pole_params) and np.all(pole_params <= max_params)
    return inside_bounds


def get_nnsc_pred(pole_class, grid_x, data_y, 
                  re_max, re_min, im_max, im_min, 
                  coeff_re_max, coeff_re_min, 
                  coeff_im_max, coeff_im_min,
                  model_path, 
                  with_bounds=False, method='lm', maxfev=100000, 
                  num_tries=1, xtol = 1e-8):
    '''
    Find optimal parameters for fitting data to a given Pole Configuration; 
    First get predictions from a NN Regressor, then give them as p0 to scipy curve_fit

    pole_class: int = 0-8 
        The Class of the Pole Configuration    

    grid_x, data_y: ndarray of shape (n,) or (1,n)
        Points to be used for fitting
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Regressor.
            
            There must additionally be a subdirectory called 'data' containing the standardization files called 
            "variances.npy" for the inputs and "variances_params.npy", "means_params.npy" for the outputs.
            
    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method
        
    maxfev: int > 0 , default=100000
        Maximal number of function evaluations (see SciPy's curve_fit)
        
    num_tries: int > 0, default=1
        The number of times the fit shall be tried (with varying initial guesses)
        
    xtol: float or list of floats, default 1e-8
        Convergence criterion (see SciPy's curve_fit)
    
    returns: ndarray of shape (k,), str
        Optimized parameters of the chosen Pole Configuration/Class, info whether the params are from NNSC, SC or NN
    '''
    
    # Get NN regressor_pred (average over all models in /models)
    pred_nn   = np.mean(get_regressor_pred(data_y=data_y, model_path=model_path), axis=0)[0]

    # Try NNSC Fit
    print('Getting NNSC prediction...')
    pred = get_scipy_pred_dual(pole_class=pole_class, grid_x=grid_x, data_y=data_y, 
                               re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                               coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                               coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                               with_bounds=False, p0=pred_nn,
                               method='lm', maxfev=1000000, num_tries=1, xtol=1e-8)
    pred_type = 'NNSC'
    
    # If the NNSC fit failed or exceeds boundaries, try a SciPy fit with random p0
    if np.isnan(pred[0]) or ~check_inside_bounds(pole_class=pole_class, pole_params=pred, 
                        re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                        coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                        coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min):    
        print('NNSC failed or exceeds boundaries... Getting SciPy fit with random p0...')
        pred = get_scipy_pred_dual(pole_class=pole_class, grid_x=grid_x, data_y=data_y, 
                               re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                               coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                               coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                               with_bounds=False, p0='random',
                               method='lm', maxfev=1000000, num_tries=10, xtol=1e-8) 
        pred_type = 'SC'
        
    # If the SciPy fits with random p0 also failed or exceeds boundaries, take the ANN prediction as a solution
    if np.isnan(pred[0]) or ~check_inside_bounds(pole_class=pole_class, pole_params=pred, 
                        re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                        coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                        coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min):
        print('SciPy fit with random p0 failed or exceeds boundaries... Take NN prediction instead')
        pred = pred_nn
        pred_type = 'NN'
         
    # Add zeros for imag parts of real poles
    pred = add_zero_imag_parts_dual(pole_class=pole_class, pole_params=pred).reshape(-1)
        
    # clip parameters (only relevant if ANN exceeds bounds, which is unlikely to happen if the real parameters are within the range of the training parameters)
    lower = np.array([re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max])
    upper = np.array([re_max, im_max,  coeff_re_max,  coeff_im_max,  coeff_re_max,  coeff_im_max])
    if pole_class in [0,1]:
        None
    elif pole_class in [2,3,4]:
        lower = np.tile(lower,(2))
        upper = np.tile(upper,(2))
    elif pole_class in [5,6,7,8]:
        lower = np.tile(lower,(3))
        upper = np.tile(upper,(3))
    pred = np.minimum(pred, upper)
    pred = np.maximum(pred, lower)
    
    # Sort Poles by Abs of Pole Position
    pred = pole_config_organize_abs_dual(pole_class=pole_class, pole_params=pred.reshape(1,-1)).reshape(-1)
    
    return pred, pred_type


def get_all_nnsc_preds(grid_x, data_y,       
                       re_max, re_min, im_max, im_min, 
                       coeff_re_max, coeff_re_min, 
                       coeff_im_max, coeff_im_min,
                       model_path, regressor_subdirs,
                       with_bounds=False, method='lm', maxfev=100000, 
                       num_tries=1, xtol = 1e-8):
    '''
    Find optimal parameters for fitting data to all 9 different Pole Configurations; 
    First get predictions from a NN Regressor, then give them as p0 to scipy curve_fit
    
    grid_x, data_y: ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
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
    
    regressor_subdirs: list of str        
        list of the names of the subdirs of the individual regressors
    
    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method
        
    maxfev: int > 0 , default=100000
        Maximal number of function evaluations (see SciPy's curve_fit)
        
    num_tries: int > 0, default=1
        The number of times the fit shall be tried (with varying initial guesses)
        
    xtol: float or list of floats, default 1e-8
        Convergence criterion (see SciPy's curve_fit)

    returns: list of 9 ndarrays of shapes (k_i,), i=0...8
        Optimized parameters of the different Pole Configurations/Classes
    '''
    preds = []
    for subdir in regressor_subdirs:
        try:
            pred = get_nnsc_pred(pole_class=int(subdir[0]), 
                                 grid_x=grid_x, data_y=data_y, 
                                 re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                 coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                 coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                 model_path=os.path.join(model_path, subdir + '/'),
                                 with_bounds=with_bounds,
                                 method=method, maxfev=maxfev, num_tries=num_tries, xtol=xtol)
        except:
            pred = np.array([])
        preds.append(pred)
    return preds






