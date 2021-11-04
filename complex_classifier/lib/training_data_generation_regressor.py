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
from joblib import Parallel, delayed

from lib.get_params_functions import get_train_params
from lib.curve_calc_functions import pole_curve_calc2, pole_curve_calc
from lib.standardization_functions import std_data_new, std_data
from lib.training_data_generation_classifier import drop_small_poles_2, drop_near_poles
from lib.pole_config_organize import pole_config_organize_abs as pole_config_organize
from lib.diverse_functions import mse, drop_not_finite_rows
from lib.scipy_fit_functions import get_scipy_pred


def add_zero_imag_parts(pole_class, pole_params):
    """
    Adds columns with zero imaginary parts to real poles
    
    pole_class: int: 0-8
        The pole class
        
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be manipulated
    
    returns: numpy.ndarray of shape (m,k'), where m is the number of samples and k' depends on the pole class (e.g. k'=4 for pole_class=0)
        Manipulated pole parameters
    """
    pole_params = np.atleast_2d(pole_params)
    m = np.shape(pole_params)[0]
    zeros = np.zeros((m,1))
    
    if pole_class == 0:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 2:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 3:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2:]
                                ))
    elif pole_class == 5:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 6:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4:]
                                ))
    elif pole_class == 7:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2:]
                                ))
    return pole_params

def get_data_x_regressor(pole_class, 
               data_y, grid_x, 
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min,
               with_bounds=True,
               p0='default', method='trf', maxfev=100000, num_tries=1, xtol = 1e-8):
    '''
    Generates the unstandardized network input from the pole curves (out_re)
    
    pole_class: int: 0-8
        The pole class
    
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
    
    returns: two numpy.ndarrays of shapes (m,1+k) and (m,len(grid_x)), where k depends on the pole class
        data_x (network input) and the function values, not yet normalized
    '''
    grid_x = grid_x.reshape(-1)
    data_y = np.atleast_2d(data_y)
    
    # Get Scipy predictions for each sample
    print('Getting SciPy predictions...')
    def get_scipy_preds_tmp(data_y_fun):
        return get_scipy_pred(pole_class = pole_class, 
                              grid_x=grid_x, data_y=data_y_fun, 
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                              with_bounds=with_bounds, p0=p0,
                              method=method, maxfev=maxfev, num_tries=num_tries, xtol=xtol)
    params_fit = Parallel(n_jobs=-1, backend="loky", verbose=10)(
                 map(delayed(get_scipy_preds_tmp), list(data_y)))
    params_fit = np.array(params_fit)

    # Calculate out_re for the different predicted pole configurations
    out_re_fit = pole_curve_calc2(pole_class=pole_class, pole_params=params_fit,   grid_x=grid_x)
    '''
    Note: If SciPy fails and returns a nan, then pole_functions raises 'RuntimeWarning: invalid value encountered in true_divide'.
          We don't need to worry about this, since we just drop those samples.
    '''
    
    # Calculate the different MSEs
    mse_fit =  mse(data_y, out_re_fit,   ax=1).reshape(-1,1) 
    # Apply log10 to the MSEs to bring them to a similiar scale
    mse_fit = np.log10(mse_fit)
    # IF MSE=0 (perfect fit), then log(MSE)=-inf. Set to a finite value instead:
    mse_fit[mse_fit==-np.inf] = np.min(mse_fit[np.isfinite(mse_fit)])
        
    # Everything together then gives data_x that is used to train the classifier
    data_x = np.hstack((mse_fit, params_fit))
    return data_x, data_y


def create_training_data_regressor(mode, pole_class, 
                                   length, grid_x, 
                                   re_max, re_min, im_max, im_min, 
                                   coeff_re_max, coeff_re_min, 
                                   coeff_im_max, coeff_im_min,
                                   with_bounds, data_dir, fact, dst_min, 
                                   p0, method, maxfev, num_tries, xtol):
    '''
    Creates training data for the NN regressor
    
    mode: str: 'preparation' or 'update'
        Two different operating types:
            
        Preparation: New data is generated. It is standardized (a new standardization is created) and saved to the disk. This function returns None.
    
        Update: New data is generated and standardized using means and variances files from data_dir. It is then returned by this function.
    
    pole_class: int: 0-8
        The pole class
    
    length: int>0
        The number of samples to be generated
           
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated

    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True        

    with_bounds: list of bools
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored/ loaded from
        
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

    returns: None or 2 numpy.ndarrays of shapes (length,len(methods)*(k+1) + n) and (length,k), where n is the number of gridpoints and k depends on the pole class
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
    
    ###########################################################################
    # Get the scipy fits by fitting out_re to this specific class
    
    # Get data_x
    for i in range(len(method)): #use different SciPy fitting methods
        print('method: ', method[i])
        print('with_bounds: ', with_bounds[i])
        print('p0: ', p0[i])
        print('num_tries: ', num_tries[i])
        print('maxfev: ', maxfev[i])
        print('xtol: ', xtol[i])
        data_x_i, out_re = get_data_x_regressor(pole_class=pole_class, 
                                                data_y=out_re, grid_x=grid_x, 
                                                re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                                coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                                coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                                with_bounds=with_bounds[i], 
                                                p0=p0[i], method=method[i], maxfev=maxfev[i], num_tries=num_tries[i], xtol=xtol[i])  

        # Add zero imaginary parts for real poles
        params_i = data_x_i[:,1:]
        mse_i    = data_x_i[:,0].reshape(-1,1)
        params_i = add_zero_imag_parts(pole_class=pole_class, pole_params=params_i)
        # Sort by abs
        params_i = pole_config_organize(pole_class=pole_class, pole_params=params_i)
        data_x_i = np.hstack((mse_i, params_i))
        
        if i==0:
            data_x = data_x_i
        else:
            data_x = np.hstack([data_x, data_x_i])
        
        tmp = len(out_re)
        ## Get rid of infinities that occur if the sample couldn't be fitted
        data_x, out_re, params, params_i = drop_not_finite_rows(
                                    data_x, out_re, params, params_i)
        num_dropped = len(out_re) - tmp
        
        # calculate Params RMSE of the fitting method:
        params_rmse         = np.sqrt(np.mean((params_i - params)**2, axis=0))
        params_overall_rmse = np.sqrt(np.mean((params_i - params)**2))
        
        # write info about fits to txt file
        dictionary = repr({
                      'method': method[i],
                      'with_bounds': with_bounds[i],
                      'p0': p0[i],
                      'num_tries': num_tries[i],
                      'maxfev': maxfev[i],
                      'xtol': xtol[i],
                      'num_dropped': num_dropped,
                      'params_rmse': params_rmse,
                      'params_overall_rmse': params_overall_rmse
                      })
        f = open( 'scipy_fits_info.txt', 'a' )
        f.write( dictionary + '\n' )
        f.close()
        
    data_x = np.hstack([data_x, out_re])
    ########################################################################### 

    if mode == 'preparation':
        # Create initial data for training      
        # Standardize Inputs
        data_x = std_data_new(data=data_x, with_mean=True, name_var="variances.npy", 
                              name_mean="means.npy", std_path=data_dir) 
        #Standardize Outputs
        params = std_data_new(data=params, with_mean=True, name_var="variances_params.npy", 
                              name_mean="means_params.npy", std_path=data_dir)
        # Save training data
        np.save(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), data_x)
        np.save(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), params)
        print("Successfully saved x data of shape ", np.shape(data_x))
        print("Successfully saved y data of shape ", np.shape(params))
        return None
    
    elif mode == 'update':
        # Update dataset during training of the regressor 
        # Standardize Inputs
        data_x = std_data(data=data_x, with_mean=True, name_var="variances.npy", 
                          name_mean="means.npy", std_path=data_dir)   
        #Standardize Outputs
        params = std_data(data=params, with_mean=True, name_var="variances_params.npy", 
                              name_mean="means_params.npy", std_path=data_dir)
        return data_x, params
    
    else:
        sys.exit("Undefined mode.")





#
