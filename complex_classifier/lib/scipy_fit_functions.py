# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:09:38 2021

@author: siegfriedkaidisch

Functions, that use SciPy's curve_fit to get pole parameters

"""
import numpy as np
from scipy.optimize import curve_fit

from parameters import coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max
from lib.pole_objective_functions import objective_1r, objective_1c
from lib.pole_objective_functions import objective_2r, objective_1r1c, objective_2c
from lib.pole_objective_functions import objective_3r, objective_2r1c, objective_1r2c, objective_3c


def pole_config_organize(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Parameters of real poles are always kept at the front
        
        Poles are ordered by Re(pole position) from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params(), but with imaginary parts of real poles removed (see pole_curve_calc vs pole_curve_calc2)
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray of shape (k,m), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray of shape (k,m)
        The organized pole configurations
    '''
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
        Define a box. Parameter configurations outside this box are dropped
    
    returns: numpy.ndarray of shape (k,)
        Optimized parameters of the chosen pole class
    '''
    grid_x = np.reshape(grid_x,(-1))
    data_y = np.reshape(data_y,(-1))
    
    if pole_class == 0:
        lower = [re_min, -coeff_re_max]
        upper = [re_max, coeff_re_max]
        params_tmp, _ = curve_fit(objective_1r, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_1r, grid_x, data_y, maxfev=100000, p0=p0)
        
    elif pole_class == 1:
        lower = [re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_1c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_1c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 2:
        lower = [re_min, -coeff_re_max, re_min, -coeff_re_max]
        upper = [re_max, coeff_re_max, re_max, coeff_re_max]
        params_tmp, _ = curve_fit(objective_2r, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_2r, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 3:
        lower = [re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_1r1c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_1r1c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 4:
        lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_2c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_2c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 5:
        lower = [re_min, -coeff_re_max, re_min, -coeff_re_max, re_min, -coeff_re_max]
        upper = [re_max, coeff_re_max, re_max, coeff_re_max, re_max, coeff_re_max]
        params_tmp, _ = curve_fit(objective_3r, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_3r, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 6:
        lower = [re_min, -coeff_re_max, re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, coeff_re_max, re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_2r1c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_2r1c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
    elif pole_class == 7:
        lower = [re_min, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_1r2c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_1r2c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1)   
        
    elif pole_class == 8:
        lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max]
        upper = [re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max]
        params_tmp, _ = curve_fit(objective_3c, grid_x, data_y, maxfev=100000, bounds=(lower, upper), p0=p0) if with_bounds else \
                      curve_fit(objective_3c, grid_x, data_y, maxfev=100000, p0=p0)
        params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(-1,1)).reshape(-1) 
        
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
        try:
            params.append(get_scipy_pred(pole_class=i, grid_x=grid_x, data_y=data_y, with_bounds=with_bounds, p0=None))   
        except:
            params.append(np.array([]))
    return params


def get_all_scipy_preds_dataprep(grid_x, data_y, labels, with_bounds=False):
    '''
    Uses Scipy curve_fit to fit all 9 different pole classes onto multiple data samples for creating data to train a NN. Drops a sample, if there is an error while fitting.
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints
        
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values to be fitted
        
    labels: numpy.ndarray of shape (m,) or (m,j)
        The actual labels/pole classes corresponding to the samples; may also contain the parameters of the sample. 
        Note: This is only handed to this function, so samples that couldn't be fitted can be dropped from it accordingly.
    
    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    returns: 11 numpy.ndarrays of shapes (m,n), (m,1) or (m,j) and (k_i,m) for i=0...8, where m is the number of samples, that could successfully be fitted
        data_y, labels, optimized parameters
    '''
    grid_x = grid_x.reshape(-1)
    data_y = np.atleast_2d(data_y)
    
    params_1r   = []
    params_1c   = []
    params_2r   = []
    params_1r1c = []
    params_2c   = []
    params_3r   = []
    params_2r1c = []
    params_1r2c = []
    params_3c   = []
    new_data_y = []
    new_labels = []
    for i in range(np.shape(data_y)[0]):
        try:
            print(i)
            data_y_i = data_y[i]
            
            params_tmp_1r   = get_scipy_pred(pole_class=0, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_1c   = get_scipy_pred(pole_class=1, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_2r   = get_scipy_pred(pole_class=2, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_1r1c = get_scipy_pred(pole_class=3, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_2c   = get_scipy_pred(pole_class=4, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_3r   = get_scipy_pred(pole_class=5, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_2r1c = get_scipy_pred(pole_class=6, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_1r2c = get_scipy_pred(pole_class=7, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            params_tmp_3c   = get_scipy_pred(pole_class=8, grid_x=grid_x, data_y=data_y_i, with_bounds=with_bounds, p0=None)
            
            params_1r.append(params_tmp_1r)
            params_1c.append(params_tmp_1c)
            params_2r.append(params_tmp_2r)
            params_1r1c.append(params_tmp_1r1c)
            params_2c.append(params_tmp_2c)
            params_3r.append(params_tmp_3r)
            params_2r1c.append(params_tmp_2r1c)
            params_1r2c.append(params_tmp_1r2c)
            params_3c.append(params_tmp_3c)
            new_labels.append(labels[i])
            new_data_y.append(data_y_i)
        except:
            print('An error occured, the sample will be dropped!')
    
    new_data_y = np.array(new_data_y)
    params_1r   = np.array(params_1r).transpose()
    params_1c   = np.array(params_1c).transpose()
    params_2r   = np.array(params_2r).transpose()
    params_1r1c = np.array(params_1r1c).transpose()
    params_2c   = np.array(params_2c).transpose()
    params_3r   = np.array(params_3r).transpose()
    params_2r1c = np.array(params_2r1c).transpose()
    params_1r2c = np.array(params_1r2c).transpose()
    params_3c   = np.array(params_3c).transpose()
    new_labels  = np.array(new_labels)
    if len(new_labels.shape) == 1:
        new_labels = new_labels.reshape(-1,1)
    
    return new_data_y, new_labels, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c













