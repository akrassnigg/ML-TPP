# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:09:38 2021

@author: siegfriedkaidisch

Functions, that use SciPy's curve_fit to get pole parameters

"""
import numpy as np
from scipy.optimize import curve_fit
import time
from joblib import Parallel, delayed
import multiprocessing

from parameters import coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max
from parameters import xtol
from lib.pole_objective_functions import objective_1r, objective_1c
from lib.pole_objective_functions import objective_2r, objective_1r1c, objective_2c
from lib.pole_objective_functions import objective_3r, objective_2r1c, objective_1r2c, objective_3c
from lib.pole_objective_functions import objective_1r_jac, objective_1c_jac
from lib.pole_objective_functions import objective_2r_jac, objective_1r1c_jac, objective_2c_jac
from lib.pole_objective_functions import objective_3r_jac, objective_2r1c_jac, objective_1r2c_jac, objective_3c_jac


def pole_config_organize(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Parameters of real poles are always kept at the front
        
        Poles are ordered by Re(pole position) from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params(), but with imaginary parts of real poles removed (see pole_curve_calc vs pole_curve_calc2)
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray of shape (m,k)
        The organized pole configurations
    '''
    pole_params = pole_params.transpose()
    if pole_class == 0:
        None
        
    elif pole_class == 1:
        #make sure that Im(Pole-position)>0):
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            if params_tmp[1]   < 0:
                params_tmp[1] *= -1
                params_tmp[3] *= -1 
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 2:
        #Order poles by Re(Pole-position)
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp = [params_tmp[0:2],params_tmp[2:4]]
            params_tmp = np.reshape(sorted(params_tmp, key=lambda paramsi: paramsi[0]), (-1))
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 3:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            if params_tmp[3] < 0:
                params_tmp[3] *= -1
                params_tmp[5] *= -1 
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 4:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp = [params_tmp[0:4],params_tmp[4:8]]
            params_tmp = np.reshape(sorted(params_tmp, key=lambda paramsi: paramsi[0]), (-1))
            if params_tmp[1] < 0:
                params_tmp[1] *= -1
                params_tmp[3] *= -1 
            if params_tmp[5] < 0:
                params_tmp[5] *= -1
                params_tmp[7] *= -1
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 5:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp = [params_tmp[0:2],params_tmp[2:4], params_tmp[4:6]]
            params_tmp = np.reshape(sorted(params_tmp, key=lambda paramsi: paramsi[0]), (-1))
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 6:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp_r = [params_tmp[0:2],params_tmp[2:4]]
            params_tmp_r = np.reshape(sorted(params_tmp_r, key=lambda paramsi: paramsi[0]), (-1))
            params_tmp_c = params_tmp[4:8]
            params_tmp =  np.hstack((params_tmp_r, params_tmp_c))
            if params_tmp[5] < 0:
                params_tmp[5] *= -1
                params_tmp[7] *= -1 
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 7:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp_c = [params_tmp[2:6],params_tmp[6:10]]
            params_tmp_c = np.reshape(sorted(params_tmp_c, key=lambda paramsi: paramsi[0]), (-1))
            params_tmp_r = params_tmp[0:2]
            params_tmp =  np.hstack((params_tmp_r, params_tmp_c))
            if params_tmp[3] < 0:
                params_tmp[3] *= -1
                params_tmp[5] *= -1 
            if params_tmp[7] < 0:
                params_tmp[7] *= -1
                params_tmp[9] *= -1   
            pole_params[:,i]   = params_tmp
        
    elif pole_class == 8:
        for i in range(np.shape(pole_params)[1]):   # for each pole configuration
            params_tmp         = pole_params[:,i]
            params_tmp = [params_tmp[0:4],params_tmp[4:8], params_tmp[8:12]]
            params_tmp = np.reshape(sorted(params_tmp, key=lambda paramsi: paramsi[0]), (-1))
            if params_tmp[1] < 0:
                params_tmp[1] *= -1
                params_tmp[3] *= -1 
            if params_tmp[5] < 0:
                params_tmp[5] *= -1
                params_tmp[7] *= -1  
            if params_tmp[9] < 0:
                params_tmp[9] *= -1
                params_tmp[11] *= -1  
            pole_params[:,i]   = params_tmp
    pole_params = pole_params.transpose()
    return pole_params

def get_scipy_pred(pole_class, grid_x, data_y, with_bounds=False, p0=None,
                   re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                   coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                   coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min):
    '''
    Uses Scipy curve_fit to fit different pole classes onto single (!) data sample
    
    pole_class: int = 0-8 
        The class of the pole configuration to be found
    
    grid_x, data_y: numpy.ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    p0: list or numpy.ndarray of shape (k,), default=None
        Initial guesses for parameter search
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric, defaults read from parameters file
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    returns: numpy.ndarray of shape (k,)
        Optimized parameters of the chosen pole class or nans if the fit failed
    '''
    #with_bounds = True
    method      = 'trf'
    #xtol        = 1e-8
    maxfev      = 100000
    
    grid_x = np.reshape(grid_x,(-1))
    data_y = np.reshape(data_y,(-1))
    
    if isinstance(xtol, list):
        xtol0, xtol1, xtol2, xtol3, xtol4, xtol5, xtol6, xtol7, xtol8 = xtol
    else:
        xtol0, xtol1, xtol2, xtol3, xtol4, xtol5, xtol6, xtol7, xtol8 = [xtol for i in range(9)]
    
    if pole_class == 0:
        try:
            lower = [re_min, -coeff_re_max]
            upper = [re_max, coeff_re_max]
            params_tmp, _ = curve_fit(objective_1r, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_1r_jac, xtol=xtol0, method=method) if with_bounds else \
                          curve_fit(objective_1r, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_1r_jac, xtol=xtol0, method=method)
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(2)])
        
    elif pole_class == 1:
        try:
            lower = [re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_1c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_1c_jac, xtol=xtol1, method=method) if with_bounds else \
                          curve_fit(objective_1c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_1c_jac, xtol=xtol1, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(4)])
            
    elif pole_class == 2:
        try:
            lower = [re_min, -coeff_re_max, re_min, -coeff_re_max]
            upper = [re_max, coeff_re_max, re_max, coeff_re_max]
            params_tmp, _ = curve_fit(objective_2r, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_2r_jac, xtol=xtol2, method=method) if with_bounds else \
                          curve_fit(objective_2r, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_2r_jac, xtol=xtol2, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(4)])
            
    elif pole_class == 3:
        try:
            lower = [re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_1r1c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_1r1c_jac, xtol=xtol3, method=method) if with_bounds else \
                          curve_fit(objective_1r1c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_1r1c_jac, xtol=xtol3, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(6)])
            
    elif pole_class == 4:
        try:
            lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_2c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_2c_jac, xtol=xtol4, method=method) if with_bounds else \
                          curve_fit(objective_2c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_2c_jac, xtol=xtol4, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(8)])
            
    elif pole_class == 5:
        try:
            lower = [re_min, -coeff_re_max, re_min, -coeff_re_max, re_min, -coeff_re_max]
            upper = [re_max, coeff_re_max, re_max, coeff_re_max, re_max, coeff_re_max]
            params_tmp, _ = curve_fit(objective_3r, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_3r_jac, xtol=xtol5, method=method) if with_bounds else \
                          curve_fit(objective_3r, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_3r_jac, xtol=xtol5, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(6)])
            
    elif pole_class == 6:
        try:
            lower = [re_min, -coeff_re_max, re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, coeff_re_max, re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_2r1c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_2r1c_jac, xtol=xtol6, method=method) if with_bounds else \
                          curve_fit(objective_2r1c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_2r1c_jac, xtol=xtol6, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(8)])
            
    elif pole_class == 7:
        try:
            lower = [re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_1r2c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_1r2c_jac, xtol=xtol7, method=method) if with_bounds else \
                          curve_fit(objective_1r2c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_1r2c_jac, xtol=xtol7, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1)   
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(10)])
            
    elif pole_class == 8:
        try:
            lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
            upper = [re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
            params_tmp, _ = curve_fit(objective_3c, grid_x, data_y, maxfev=maxfev, bounds=(lower, upper), p0=p0, jac=objective_3c_jac, xtol=xtol8, method=method) if with_bounds else \
                          curve_fit(objective_3c, grid_x, data_y, maxfev=maxfev, p0=p0, jac=objective_3c_jac, xtol=xtol8, method=method)
            params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
        except:
            print('An error occured, the sample will be dropped!')
            params_tmp = np.array([np.nan for i in range(12)])
            
    return params_tmp


def get_all_scipy_preds(grid_x, data_y, with_bounds=False):
    '''
    Uses Scipy curve_fit to fit all 9 different pole classes onto a single data sample
    
    grid_x, data_y: numpy.ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    returns: list of 9 numpy.ndarrays of shapes (k_i,), i=0...8
        Optimized parameters of the different pole classes
    '''
    params = []
    for i in range(9):
        params.append(get_scipy_pred(pole_class=i, grid_x=grid_x, data_y=data_y, with_bounds=with_bounds, p0=None))   
    return params


def get_all_scipy_preds_dataprep(grid_x, data_y, with_bounds=False):
    '''
    Uses Scipy curve_fit to fit all 9 different pole classes onto multiple data samples for creating data to train a NN. 
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints
        
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values to be fitted

    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    returns: 9 numpy.ndarrays of shapes (m,k_i) for i=0...8, where m is the number of samples
        optimized parameters (nans if the fit failed)
    '''
    grid_x = grid_x.reshape(-1)
    data_y = np.atleast_2d(data_y)

    def get_all_scipy_preds_tmp(data_y_fun):
        return get_all_scipy_preds(grid_x=grid_x, data_y=data_y_fun, with_bounds=with_bounds)
    
    params_tmp = Parallel(n_jobs=-1, backend="loky", verbose=10)(
                 map(delayed(get_all_scipy_preds_tmp), list(data_y)))
   
    params_1r   = [tmp[0] for tmp in params_tmp]
    params_1c   = [tmp[1] for tmp in params_tmp]
    params_2r   = [tmp[2] for tmp in params_tmp]
    params_1r1c = [tmp[3] for tmp in params_tmp]
    params_2c   = [tmp[4] for tmp in params_tmp]
    params_3r   = [tmp[5] for tmp in params_tmp]
    params_2r1c = [tmp[6] for tmp in params_tmp]
    params_1r2c = [tmp[7] for tmp in params_tmp]
    params_3c   = [tmp[8] for tmp in params_tmp]
 
    params_1r   = np.array(params_1r)
    params_1c   = np.array(params_1c)
    params_2r   = np.array(params_2r)
    params_1r1c = np.array(params_1r1c)
    params_2c   = np.array(params_2c)
    params_3r   = np.array(params_3r)
    params_2r1c = np.array(params_2r1c)
    params_1r2c = np.array(params_1r2c)
    params_3c   = np.array(params_3c)

    return params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c













