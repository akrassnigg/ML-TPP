#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:25:53 2021

Transferred integration-point routines written by tuh

@author: ank
"""


import scipy.special as spspecial
from scipy import exp, pi, log, sqrt, real, imag, power



def mom_map(NumPar):
    """
    mom_map(x, NumPar)
    
    Maps the Gauss integration points (-1,1) to (0,infinity).
    Returns mapped array of mapping and derivative w.r.t. x as tuple
    ( p(x), dp(x)/dx ).
    
    - 'a' is modified according to 'SupMom' for 'graz'-mapping
    - if 'sup_mom'=NumPar['SupMom'] is given, 'a' is changed such that maximal momentum is met
    """
    c_0     = NumPar['c_l_rdl']
    a       = NumPar['a_l_rdl'] if 'a_l_rdl' in NumPar.keys() else 1
    sup_mom = NumPar['SupMom'] if 'SupMom' in NumPar.keys() else False
    
    x, w = spspecial.p_roots(NumPar['n_l_rdl'])#Gauss-Legendre roots+weights
    
    if NumPar['mapping']=='dubna':
        mapping     = c_0*(1+x**a)/(1-x**a)
        derivative  = 2*c_0 * (a*x**(a-1)) / (1-x**a)**2
    
    elif NumPar['mapping']=='graz':
        x_bar = (1.+x)/2
        if sup_mom: a = log(1. + sup_mom/c_0 * (exp(1)-exp(x_bar.max()))) / x_bar.max()
        mapping     = c_0 * (exp(a*x_bar)-1) / (exp(1)-exp(x_bar))
        derivative  = 0.5*( c_0*a*exp(a*x_bar) + mapping*exp(x_bar) ) / (exp(1)-exp(x_bar))
    
    return mapping, derivative, x, w


