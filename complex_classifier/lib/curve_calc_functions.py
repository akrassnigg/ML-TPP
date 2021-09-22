# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Functions, that calculate the function values of different pole configurations on the real axis

"""
import numpy as np

from lib.pole_functions import complex_conjugate_pole_pair
    

def pole_curve_calc(pole_class, pole_params, grid_x):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    NOTE: The difference between pole_curve_calc and pole_curve_calc2 is: 
        
        pole_curve_calc assumes real pole configs to still contain the imaginary parts, just set to 0
        
        pole_curve_calc2 assumes real pole configs to not contain imaginary parts + it assumes real poles to be at the front (see get_train_params)
        
        Example: [-1., 0., 0.5, 0.] vs. [-1., 0.5], where -1=Re(pole position) and 0.5=Re(pole coefficient). 
        The former is how pole_curve_calc wants real poles to be formatted, while the latter is what pole_curve_calc2 wants.
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=4 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    returns: numpy.ndarray of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = np.reshape(grid_x, (-1))  
    pole_params = pole_params.transpose()
    
    if pole_class == 0:
        params_1r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_1r[0], params_1r[1], params_1r[2], params_1r[3])
    elif pole_class == 1:
        params_1c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_1c[0], params_1c[1], params_1c[2], params_1c[3])
    elif pole_class == 2:
        params_2r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_2r[0], params_2r[1], params_2r[2], params_2r[3]) + complex_conjugate_pole_pair(grid_x, params_2r[4], params_2r[5], params_2r[6], params_2r[7])
    elif pole_class == 3:
        params_1r1c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_1r1c[0], params_1r1c[1], params_1r1c[2], params_1r1c[3]) + complex_conjugate_pole_pair(grid_x, params_1r1c[4], params_1r1c[5], params_1r1c[6], params_1r1c[7])
    elif pole_class == 4:
        params_2c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_2c[0], params_2c[1], params_2c[2], params_2c[3]) + complex_conjugate_pole_pair(grid_x, params_2c[4], params_2c[5], params_2c[6], params_2c[7])
    elif pole_class == 5:
        params_3r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_3r[0], params_3r[1], params_3r[2], params_3r[3]) + complex_conjugate_pole_pair(grid_x, params_3r[4], params_3r[5], params_3r[6], params_3r[7]) + complex_conjugate_pole_pair(grid_x, params_3r[8], params_3r[9], params_3r[10], params_3r[11])
    elif pole_class == 6:
        params_2r1c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_2r1c[0], params_2r1c[1], params_2r1c[2], params_2r1c[3]) + complex_conjugate_pole_pair(grid_x, params_2r1c[4], params_2r1c[5], params_2r1c[6], params_2r1c[7]) + complex_conjugate_pole_pair(grid_x, params_2r1c[8], params_2r1c[9], params_2r1c[10], params_2r1c[11])
    elif pole_class == 7:
        params_1r2c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_1r2c[0], params_1r2c[1], params_1r2c[2], params_1r2c[3]) + complex_conjugate_pole_pair(grid_x, params_1r2c[4], params_1r2c[5], params_1r2c[6], params_1r2c[7]) + complex_conjugate_pole_pair(grid_x, params_1r2c[8], params_1r2c[9], params_1r2c[10], params_1r2c[11])
    elif pole_class == 8:
        params_3c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_3c[0], params_3c[1], params_3c[2], params_3c[3]) + complex_conjugate_pole_pair(grid_x, params_3c[4], params_3c[5], params_3c[6], params_3c[7]) + complex_conjugate_pole_pair(grid_x, params_3c[8], params_3c[9], params_3c[10], params_3c[11])
    return curve_pred


def pole_curve_calc2(pole_class, pole_params, grid_x):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    NOTE: The difference between pole_curve_calc and pole_curve_calc2 is: 
        
        pole_curve_calc assumes real pole configs to still contain the imaginary parts, just set to 0
        
        pole_curve_calc2 assumes real pole configs to not contain imaginary parts + it assumes real poles to be at the front (see get_train_params)
        
        Example: [-1., 0., 0.5, 0.] vs. [-1., 0.5], where -1=Re(pole position) and 0.5=Re(pole coefficient). 
        The former is how pole_curve_calc wants real poles to be formatted, while the latter is what pole_curve_calc2 wants.
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    returns: numpy.ndarray of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = np.reshape(grid_x, (-1))  
    pole_params = pole_params.transpose()
    
    if pole_class == 0:
        params_1r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_1r[0], params_1r[0]*0, params_1r[1], params_1r[0]*0)
    elif pole_class == 1:
        params_1c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_1c[0], params_1c[1], params_1c[2], params_1c[3])
    elif pole_class == 2:
        params_2r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_2r[0], params_2r[0]*0, params_2r[1], params_2r[0]*0) + complex_conjugate_pole_pair(grid_x, params_2r[2], params_2r[0]*0, params_2r[3], params_2r[0]*0)
    elif pole_class == 3:
        params_1r1c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_1r1c[0], params_1r1c[0]*0, params_1r1c[1], params_1r1c[0]*0) + complex_conjugate_pole_pair(grid_x, params_1r1c[2], params_1r1c[3], params_1r1c[4], params_1r1c[5])
    elif pole_class == 4:
        params_2c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_2c[0], params_2c[1], params_2c[2], params_2c[3]) + complex_conjugate_pole_pair(grid_x, params_2c[4], params_2c[5], params_2c[6], params_2c[7])
    elif pole_class == 5:
        params_3r = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_3r[0], params_3r[0]*0, params_3r[1], params_3r[0]*0) + complex_conjugate_pole_pair(grid_x, params_3r[2], params_3r[0]*0, params_3r[3], params_3r[0]*0) + complex_conjugate_pole_pair(grid_x, params_3r[4], params_3r[0]*0, params_3r[5], params_3r[0]*0)
    elif pole_class == 6:
        params_2r1c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_2r1c[0], params_2r1c[0]*0, params_2r1c[1], params_2r1c[0]*0) + complex_conjugate_pole_pair(grid_x, params_2r1c[2], params_2r1c[0]*0, params_2r1c[3], params_2r1c[0]*0) + complex_conjugate_pole_pair(grid_x, params_2r1c[4], params_2r1c[5], params_2r1c[6], params_2r1c[7])
    elif pole_class == 7:
        params_1r2c = pole_params
        curve_pred = complex_conjugate_pole_pair(grid_x, params_1r2c[0], params_1r2c[0]*0, params_1r2c[1], params_1r2c[0]*0) + complex_conjugate_pole_pair(grid_x, params_1r2c[2], params_1r2c[3], params_1r2c[4], params_1r2c[5]) + complex_conjugate_pole_pair(grid_x, params_1r2c[6], params_1r2c[7], params_1r2c[8], params_1r2c[9])
    elif pole_class == 8:
        params_3c = pole_params
        curve_pred   = complex_conjugate_pole_pair(grid_x, params_3c[0], params_3c[1], params_3c[2], params_3c[3]) + complex_conjugate_pole_pair(grid_x, params_3c[4], params_3c[5], params_3c[6], params_3c[7]) + complex_conjugate_pole_pair(grid_x, params_3c[8], params_3c[9], params_3c[10], params_3c[11])
    return curve_pred















