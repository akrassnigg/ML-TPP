# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:34:08 2021

@author: siegfriedkaidisch

Pole objective functions to be used by SciPy's curve_fit

"""
import numpy as np


def single_complex_pole_obj(x_re, x_im, pos_re, pos_im, coeff_re, coeff_im, power=1):
    """
    Computes the function value on a single complex point of a 
    single singularity of power power, with a multiplicative complex coefficient 
    and a position in the complex plane given by pos_re and pos_im
    
    x_re, x_im: number
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
        
    power: number
        The power of the pole
        
    returns: two numbers
        The real and imaginary parts of the function value
    """
    diff = (x_re - pos_re) + 1j * (x_im - pos_im)
    
    result = (coeff_re + 1j * coeff_im) / diff**power
    
    result_re = np.real(result)
    
    result_im = np.imag(result)
    
    return result_re, result_im


def complex_conjugate_pole_pair_obj(x_re, pos_re, pos_im, coeff_re, coeff_im, power=1):
    """
    Computes the function value on a single real point of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by pos_re and on the imaginary axis given by pos_im 
    
    x_re: number
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
    
    power: number
        The power of the pole
        
    returns: number
        The real part of the function value
    """
    result_re_plus, _ = single_complex_pole_obj(x_re, 0.0, pos_re, pos_im, coeff_re, coeff_im, power=power)
    result_re_minus, _ = single_complex_pole_obj(x_re, 0.0, pos_re=pos_re, pos_im=-pos_im, coeff_re=coeff_re, coeff_im=-coeff_im, power=power)
    
    return result_re_plus + result_re_minus


def objective_1r(x, a, c):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    return y1 


def objective_1c(x, a, b, c, d):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: number
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    return y1 


def objective_2r(x, a, c, e, g):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    f = 0
    h = 0
    
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    return y1 + y2


def objective_1r1c(x, a, c, e, f, g, h):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    return y1 + y2


def objective_2c(x, a, b, c, d, e, f, g, h):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: number
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    return y1 + y2


def objective_3r(x, a, c, e, g, i, k):
    '''
    Objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    f = 0
    h = 0
    j = 0
    l = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    y3 = complex_conjugate_pole_pair_obj(x, i, j, k, l)
    return y1 + y2 + y3


def objective_2r1c(x, a, c, e, g, i, j, k, l):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    f = 0
    h = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    y3 = complex_conjugate_pole_pair_obj(x, i, j, k, l)
    return y1 + y2 + y3


def objective_1r2c(x, a, c, e, f, g, h, i, j, k, l):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number
        The real part of the function value
    '''
    b = 0
    d = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    y3 = complex_conjugate_pole_pair_obj(x, i, j, k, l)
    return y1 + y2 + y3


def objective_3c(x, a, b, c, d, e, f, g, h, i, j, k, l):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    x: number
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    y3 = complex_conjugate_pole_pair_obj(x, i, j, k, l)
    return y1 + y2 + y3




