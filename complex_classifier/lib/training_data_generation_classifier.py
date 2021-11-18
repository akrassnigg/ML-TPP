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

from lib.get_params_functions import get_train_params_dual
from lib.scipy_fit_functions import get_all_scipy_preds_dataprep_dual
from lib.curve_calc_functions import pole_curve_calc_dual, pole_curve_calc2_dual
from lib.diverse_functions import mse, drop_not_finite_rows
from lib.standardization_functions import rm_std_data, std_data_new


def get_data_x(data_y, grid_x, 
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min,
               with_bounds=True,
               p0='random', method='lm', maxfev=1000000, num_tries=10, xtol = 1e-8):
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
    params_3r, params_2r1c, params_1r2c, params_3c = get_all_scipy_preds_dataprep_dual(grid_x=grid_x, data_y=data_y, 
                                                                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                                                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                                                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                                                                  with_bounds=with_bounds, p0=p0,
                                                                                  method=method, maxfev=maxfev, num_tries=num_tries, 
                                                                                  xtol=xtol)

    # Calculate out_re for the different predicted pole configurations
    out_re_1r   = pole_curve_calc2_dual(pole_class=0, pole_params=params_1r,   grid_x=grid_x)
    out_re_1c   = pole_curve_calc2_dual(pole_class=1, pole_params=params_1c,   grid_x=grid_x)
    out_re_2r   = pole_curve_calc2_dual(pole_class=2, pole_params=params_2r,   grid_x=grid_x)
    out_re_1r1c = pole_curve_calc2_dual(pole_class=3, pole_params=params_1r1c, grid_x=grid_x)
    out_re_2c   = pole_curve_calc2_dual(pole_class=4, pole_params=params_2c,   grid_x=grid_x)
    out_re_3r   = pole_curve_calc2_dual(pole_class=5, pole_params=params_3r,   grid_x=grid_x)
    out_re_2r1c = pole_curve_calc2_dual(pole_class=6, pole_params=params_2r1c, grid_x=grid_x)
    out_re_1r2c = pole_curve_calc2_dual(pole_class=7, pole_params=params_1r2c, grid_x=grid_x)
    out_re_3c   = pole_curve_calc2_dual(pole_class=8, pole_params=params_3c,   grid_x=grid_x)
    
    '''
    Note: If SciPy fails and returns a nan, then pole_functions raises 'RuntimeWarning: invalid value encountered in true_divide'.
          We don't need to worry about this, since we just drop those samples.
    '''

    # Calculate the different MSEs
    mse_1r    = mse(data_y, out_re_1r,   ax=1).reshape(-1,1) 
    mse_1c    = mse(data_y, out_re_1c,   ax=1).reshape(-1,1) 
    mse_2r    = mse(data_y, out_re_2r,   ax=1).reshape(-1,1) 
    mse_1r1c  = mse(data_y, out_re_1r1c, ax=1).reshape(-1,1)
    mse_2c    = mse(data_y, out_re_2c,   ax=1).reshape(-1,1) 
    mse_3r    = mse(data_y, out_re_3r,   ax=1).reshape(-1,1) 
    mse_2r1c  = mse(data_y, out_re_2r1c, ax=1).reshape(-1,1) 
    mse_1r2c  = mse(data_y, out_re_1r2c, ax=1).reshape(-1,1) 
    mse_3c    = mse(data_y, out_re_3c,   ax=1).reshape(-1,1) 
    
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
    
    # IF MSE=0 (perfect fit), then log(MSE)=-inf. Set to a finite value instead:
    mse_1r[mse_1r==-np.inf]     = np.min(mse_1r[np.isfinite(mse_1r)])
    mse_1c[mse_1c==-np.inf]     = np.min(mse_1c[np.isfinite(mse_1c)])
    mse_2r[mse_2r==-np.inf]     = np.min(mse_2r[np.isfinite(mse_2r)])
    mse_1r1c[mse_1r1c==-np.inf] = np.min(mse_1r1c[np.isfinite(mse_1r1c)])
    mse_2c[mse_2c==-np.inf]     = np.min(mse_2c[np.isfinite(mse_2c)])
    mse_3r[mse_3r==-np.inf]     = np.min(mse_3r[np.isfinite(mse_3r)])
    mse_2r1c[mse_2r1c==-np.inf] = np.min(mse_2r1c[np.isfinite(mse_2r1c)])
    mse_1r2c[mse_1r2c==-np.inf] = np.min(mse_1r2c[np.isfinite(mse_1r2c)])
    mse_3c[mse_3c==-np.inf]     = np.min(mse_3c[np.isfinite(mse_3c)])
    
    # Everything together then gives data_x that is used to train the classifier
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c))
    return data_x, data_y

def create_training_data_classifier(length, grid_x, 
                                    re_max, re_min, im_max, im_min, 
                                    coeff_re_max, coeff_re_min, 
                                    coeff_im_max, coeff_im_min,
                                    with_bounds, data_dir, 
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
        params = get_train_params_dual(pole_class=pole_class, m=nums[pole_class],
                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        out_re.append(pole_curve_calc_dual(pole_class=pole_class, pole_params=params, grid_x=grid_x))
        
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
        print('method: ', method[i])
        print('with_bounds: ', with_bounds[i])
        print('p0: ', p0[i])
        print('num_tries: ', num_tries[i])
        print('maxfev: ', maxfev[i])
        print('xtol: ', xtol[i])
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
        
        ## Get rid of infinities that occur if the sample couldn't be fitted
        data_x, out_re, labels_and_params = drop_not_finite_rows(
                                    data_x, out_re, labels_and_params)
        
        labels = labels_and_params[:,0].reshape(-1)
    
    if len(method) == 0:
        data_x = out_re
    else:   
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
