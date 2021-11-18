# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:34:08 2021

@author: siegfriedkaidisch

Pole objective functions to be used by SciPy's curve_fit

"""
import numpy as np


def single_complex_pole_obj_old(x_re, x_im, pos_re, pos_im, coeff_re, coeff_im, power=1):
    """
    Computes the function value on a single complex point of a 
    single singularity of power power, with a multiplicative complex coefficient 
    and a position in the complex plane given by pos_re and pos_im
    
    x_re, x_im: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
        
    power: number
        The power of the pole
        
    returns: two numbers or np.ndarrays of shapes (n,)
        The real and imaginary parts of the function value
    """
    diff = (x_re - pos_re) + 1j * (x_im - pos_im)
    
    result = (coeff_re + 1j * coeff_im) / diff**power
    
    result_re = np.real(result)
    
    result_im = np.imag(result)
    
    return result_re, result_im

def complex_conjugate_pole_pair_obj_old(x_re, pos_re, pos_im, coeff_re, coeff_im, power=1):
    """
    Computes the function value on a single real point of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by pos_re and on the imaginary axis given by pos_im 
    
    x_re: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
    
    power: number
        The power of the pole
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    """
    result_re_plus, _ = single_complex_pole_obj_old(x_re, 0.0, pos_re, pos_im, coeff_re, coeff_im, power=power)
    result_re_minus, _ = single_complex_pole_obj_old(x_re, 0.0, pos_re=pos_re, pos_im=-pos_im, coeff_re=coeff_re, coeff_im=-coeff_im, power=power)

    return result_re_plus + result_re_minus

##############################################################################
##############################################################################
##############################################################################

def single_complex_pole_obj(x_re, pos_re, pos_im, coeff_re, coeff_im):
    """
    Computes the function value on a single real point of a 
    single singularity of power 1, with a multiplicative complex coefficient 
    and a position in the complex plane given by pos_re and pos_im
    
    This is a faster and simplified version of single_complex_pole_obj.
    
    x_re: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
        
    returns: two numbers or np.ndarrays of shapes (n,)
        The real and imaginary parts of the function value
    """
    
    d1 = (x_re - pos_re)
    d2 = (-pos_im)
    result_re = (coeff_re*d1 + coeff_im*d2) / (d1**2 + d2**2)

    return result_re

def complex_conjugate_pole_pair_obj(x_re, pos_re, pos_im, coeff_re, coeff_im):
    """
    Computes the function value on a single real point of a
    singularity of power 1 plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by pos_re and on the imaginary axis given by pos_im 
    
    This is a faster and simplified version of complex_conjugate_pole_pair_obj_old.
    
    x_re: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    pos_re, pos_im: number
        Pole position
        
    coeff_re, coeff_im: number
        Pole coefficient
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    """
    result_re_plus = single_complex_pole_obj(x_re, pos_re, pos_im, coeff_re, coeff_im)
    
    return 2*result_re_plus

##############################################################################
##############################################################################
##############################################################################

def objective_1r(x, a, c):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    '''
    b = 0
    d = 0
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    return y1 

def objective_1c(x, a, b, c, d):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    return y1 


def objective_2r(x, a, c, e, g):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: number or np.ndarray of shape (n,)
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
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
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
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    return y1 + y2


def objective_3r(x, a, c, e, g, i, k):
    '''
    Objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
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
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
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
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
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
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (n,)
        The real part of the function value
    '''
    y1 = complex_conjugate_pole_pair_obj(x, a, b, c, d)
    y2 = complex_conjugate_pole_pair_obj(x, e, f, g, h)
    y3 = complex_conjugate_pole_pair_obj(x, i, j, k, l)
    return y1 + y2 + y3


def objective_1r_jac(x, a, c):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    denominator = (x-a)**2 
    
    da = 2 * ( -c/denominator + (c*(x-a))*(2*x-2*a)/(denominator**2) )
    dc = 2 * (x-a)/denominator
    
    jacmat = np.stack([da,dc], axis=1)
    
    return jacmat

def objective_1c_jac(x, a, b, c, d):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    denominator = (x-a)**2 + b**2 
    
    da = 2 * ( -c/denominator + (c*(x-a) -d*b)*(2*x-2*a)/(denominator**2) )
    db = 2 * ( -d/denominator + (c*(x-a) -d*b)*(   -2*b)/(denominator**2) )
    dc = 2 * (x-a)/denominator
    dd = 2 * ( -b)/denominator
    
    jacmat = np.stack([da,db,dc,dd], axis=1)
    
    return jacmat

def objective_2r_jac(x, a, c, e, g):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac(x, a, c)
    jac2   = objective_1r_jac(x, e, g)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_1r1c_jac(x, a, c, e, f, g, h):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac(x, a, c)
    jac2   = objective_1c_jac(x, e, f, g, h)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_2c_jac(x, a, b, c, d, e, f, g, h):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac(x, a, b, c, d)
    jac2   = objective_1c_jac(x, e, f, g, h)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_3r_jac(x, a, c, e, g, i, k):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac(x, a, c)
    jac2   = objective_1r_jac(x, e, g)
    jac3   = objective_1r_jac(x, i, k)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_2r1c_jac(x, a, c, e, g, i, j, k, l):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac(x, a, c)
    jac2   = objective_1r_jac(x, e, g)
    jac3   = objective_1c_jac(x, i, j, k, l)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_1r2c_jac(x, a, c, e, f, g, h, i, j, k, l):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac(x, a, c)
    jac2   = objective_1c_jac(x, e, f, g, h)
    jac3   = objective_1c_jac(x, i, j, k, l)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_3c_jac(x, a, b, c, d, e, f, g, h, i, j, k, l):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    x: number or numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (k,) or (n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac(x, a, b, c, d)
    jac2   = objective_1c_jac(x, e, f, g, h)
    jac3   = objective_1c_jac(x, i, j, k, l)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


##############################################################################
##############################################################################
##############################################################################

def objective_1r_dual(x, a, c1, c2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c1,2: number
        Real part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r(x=x, a=a, c=c1)
    y2 = objective_1r(x=x, a=a, c=c2)
    return np.hstack((y1,y2))

def objective_1c_dual(x, a, b, c1, d1, c2, d2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1c(x, a, b, c1, d1)
    y2 = objective_1c(x, a, b, c2, d2)
    return np.hstack((y1,y2))


def objective_2r_dual(x, a, c1, c2, e, g1, g2):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2r(x, a, c1, e, g1)
    y2 = objective_2r(x, a, c2, e, g2)
    return np.hstack((y1,y2))


def objective_1r1c_dual(x, a, c1, c2, e, f, g1, h1, g2, h2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r1c(x, a, c1, e, f, g1, h1)
    y2 = objective_1r1c(x, a, c2, e, f, g2, h2)
    return np.hstack((y1,y2))


def objective_2c_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2c(x, a, b, c1, d1, e, f, g1, h1)
    y2 = objective_2c(x, a, b, c2, d2, e, f, g2, h2)
    return np.hstack((y1,y2))


def objective_3r_dual(x, a, c1, c2, e, g1, g2, i, k1, k2):
    '''
    Objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_3r(x, a, c1, e, g1, i, k1)
    y2 = objective_3r(x, a, c2, e, g2, i, k2)
    return np.hstack((y1,y2))

def objective_2r1c_dual(x, a, c1, c2, e, g1, g2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2r1c(x, a, c1, e, g1, i, j, k1, l1)
    y2 = objective_2r1c(x, a, c2, e, g2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_1r2c_dual(x, a, c1, c2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r2c(x, a, c1, e, f, g1, h1, i, j, k1, l1)
    y2 = objective_1r2c(x, a, c2, e, f, g2, h2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_3c_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
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
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_3c(x, a, b, c1, d1, e, f, g1, h1, i, j, k1, l1)
    y2 = objective_3c(x, a, b, c2, d2, e, f, g2, h2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_1r_jac_dual(x, a, c1, c2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    zeros = np.zeros((len(x),1))
    
    jac1 = objective_1r_jac(x, a, c1)
    jac2 = objective_1r_jac(x, a, c2)
    
    jac1 = np.hstack([jac1, zeros])
    jac2 = np.hstack([jac2[:,0].reshape(-1,1), zeros, jac2[:,1].reshape(-1,1)])
    
    jacmat = np.vstack([jac1, jac2])
    return jacmat

def objective_1c_jac_dual(x, a, b, c1, d1, c2, d2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)        
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    zeros = np.zeros((len(x),2))
    
    jac1 = objective_1c_jac(x, a, b, c1, d1)
    jac2 = objective_1c_jac(x, a, b, c2, d2)
    
    jac1 = np.hstack([jac1, zeros])
    jac2 = np.hstack([jac2[:,0:2], zeros, jac2[:,2:]])
    
    jacmat = np.vstack([jac1, jac2])
    return jacmat

def objective_2r_jac_dual(x, a, c1, c2, e, g1, g2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, g2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_1r1c_jac_dual(x, a, c1, c2, e, f, g1, h1, g2, h2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_2c_jac_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac_dual(x, a, b, c1, d1, c2, d2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_3r_jac_dual(x, a, c1, c2, e, g1, g2, i, k1, k2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)          
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, g2)
    jac3   = objective_1r_jac_dual(x, i, k1, k2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_2r1c_jac_dual(x, a, c1, c2, e, g1, g2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, c2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_1r2c_jac_dual(x, a, c1, c2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)        
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_3c_jac_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
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
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac_dual(x, a, b, c1, d1, c2, d2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat



















