# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:03:50 2021

@author: siegfriedkaidisch

Pole functions written in Torch
"""
import torch


def single_complex_pole_torch(x_re, x_im, part_re, part_im, coeff_re, coeff_im, device, power=1):
    """
    Computes the function values on a list of complex coordinates of a 
    single singularity of power power, with a multiplicative complex coefficient 
    and a position in the complex plane given by part_re and part_im
    
    x_re, x_im: torch.Tensor of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: torch.Tensor of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: torch.Tensor of shape (m,) or (1,m)
        Pole coefficients
        
    device: torch.device
        Where (on which device) shall the calculation be done
        
    power: number, default=1
        The power of the poles
        
    returns: two torch.Tensors of shape (m,n)
        The real and imaginary parts of the function values
    """
    x_re     = x_re.to(device=device)
    x_im     = x_im.to(device=device)
    part_re  = part_re.to(device=device)
    part_im  = part_im.to(device=device)
    coeff_re = coeff_re.to(device=device)
    coeff_im = coeff_im.to(device=device)
    
    x_re = torch.reshape(x_re, (-1,))
    x_im = torch.reshape(x_im, (-1,))
    part_re = torch.reshape(part_re, (-1,))
    part_im = torch.reshape(part_im, (-1,))
    coeff_re = torch.reshape(coeff_re, (-1,))
    coeff_im = torch.reshape(coeff_im, (-1,))
 
    num_params = len(part_re)
    num_points = len(x_re)
    
    diff = ((torch.tile(x_re, (num_params,1)) - torch.tile(torch.reshape(part_re,(num_params,1)),(1, num_points))) + 
        1j * (torch.tile(x_im, (num_params,1)) - torch.tile(torch.reshape(part_im,(num_params,1)),(1, num_points))) )
    
    result = (torch.tile(torch.reshape(coeff_re,(num_params,1)),(1, num_points)) + 1j*torch.tile(torch.reshape(coeff_im,(num_params,1)),(1, num_points)))/diff**power
    result_re = torch.real(result)
    result_im = torch.imag(result)
    
    return result_re, result_im


def complex_conjugate_pole_pair_torch(x_re, part_re, part_im, coeff_re, coeff_im, device, power=1):
    """
    Computes the function values on a list of coordinates on the real axis of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by part_re and on the imaginary axis given by part_im 
    
    x_re: torch.Tensor of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: torch.Tensor of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: torch.Tensor of shape (m,) or (1,m)
        Pole coefficients
        
    device: torch.device
        Where (on which device) shall the calculation be done
        
    power: number, default=1
        The power of the poles
        
    returns: torch.Tensor of shape (m,n)
        The real part of the function values
    """
    x_re     = x_re.to(device=device)
    part_re  = part_re.to(device=device)
    part_im  = part_im.to(device=device)
    coeff_re = coeff_re.to(device=device)
    coeff_im = coeff_im.to(device=device)
    
    x_re = torch.reshape(x_re, (-1,))
    part_re = torch.reshape(part_re, (-1,))
    part_im = torch.reshape(part_im, (-1,))
    coeff_re = torch.reshape(coeff_re, (-1,))
    coeff_im = torch.reshape(coeff_im, (-1,))
    
    len_set = len(x_re)
    x_im = torch.linspace(0., 0., steps=len_set)
    x_im = x_im.to(device=device)
    
    result_re_plus, _ = single_complex_pole_torch(x_re, x_im, part_re, part_im, coeff_re, coeff_im, device=device, power=power)
    result_re_minus, _ = single_complex_pole_torch(x_re, x_im, part_re=part_re, part_im=-part_im, coeff_re=coeff_re, coeff_im=-coeff_im, device=device, power=power)

    return result_re_plus + result_re_minus


def pole_curve_calc_torch(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    NOTE: The difference between pole_curve_calc and pole_curve_calc2 is: 
        
        pole_curve_calc assumes real pole configs to still contain the imaginary parts, just set to 0
        
        pole_curve_calc2 assumes real pole configs to not contain imaginary parts + it assumes real poles to be at the front (see get_train_params)
        
        Example: [-1., 0., 0.5, 0.] vs. [-1., 0.5], where -1=Re(pole position) and 0.5=Re(pole coefficient). 
        The former is how pole_curve_calc wants real poles to be formatted, while the latter is what pole_curve_calc2 wants.
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class == 0:
        params_1r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_1r[0], params_1r[1], params_1r[2], params_1r[3], device=device)
    elif pole_class == 1:
        params_1c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_1c[0], params_1c[1], params_1c[2], params_1c[3], device=device)
    elif pole_class == 2:
        params_2r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_2r[0], params_2r[1], params_2r[2], params_2r[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r[4], params_2r[5], params_2r[6], params_2r[7], device=device)
    elif pole_class == 3:
        params_1r1c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_1r1c[0], params_1r1c[1], params_1r1c[2], params_1r1c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r1c[4], params_1r1c[5], params_1r1c[6], params_1r1c[7], device=device)
    elif pole_class == 4:
        params_2c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_2c[0], params_2c[1], params_2c[2], params_2c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2c[4], params_2c[5], params_2c[6], params_2c[7], device=device)
    elif pole_class == 5:
        params_3r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_3r[0], params_3r[1], params_3r[2], params_3r[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3r[4], params_3r[5], params_3r[6], params_3r[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3r[8], params_3r[9], params_3r[10], params_3r[11], device=device)
    elif pole_class == 6:
        params_2r1c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_2r1c[0], params_2r1c[1], params_2r1c[2], params_2r1c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r1c[4], params_2r1c[5], params_2r1c[6], params_2r1c[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r1c[8], params_2r1c[9], params_2r1c[10], params_2r1c[11], device=device)
    elif pole_class == 7:
        params_1r2c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_1r2c[0], params_1r2c[1], params_1r2c[2], params_1r2c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r2c[4], params_1r2c[5], params_1r2c[6], params_1r2c[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r2c[8], params_1r2c[9], params_1r2c[10], params_1r2c[11], device=device)
    elif pole_class == 8:
        params_3c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_3c[0], params_3c[1], params_3c[2], params_3c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3c[4], params_3c[5], params_3c[6], params_3c[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3c[8], params_3c[9], params_3c[10], params_3c[11], device=device)
    return curve_pred


def pole_curve_calc2_torch(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    NOTE: The difference between pole_curve_calc and pole_curve_calc2 is: 
        
        pole_curve_calc assumes real pole configs to still contain the imaginary parts, just set to 0
        
        pole_curve_calc2 assumes real pole configs to not contain imaginary parts + it assumes real poles to be at the front (see get_train_params)
        
        Example: [-1., 0., 0.5, 0.] vs. [-1., 0.5], where -1=Re(pole position) and 0.5=Re(pole coefficient). 
        The former is how pole_curve_calc wants real poles to be formatted, while the latter is what pole_curve_calc2 wants.
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class == 0:
        params_1r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_1r[0], params_1r[0]*0, params_1r[1], params_1r[0]*0, device=device)
    elif pole_class == 1:
        params_1c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_1c[0], params_1c[1], params_1c[2], params_1c[3], device=device)
    elif pole_class == 2:
        params_2r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_2r[0], params_2r[0]*0, params_2r[1], params_2r[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r[2], params_2r[0]*0, params_2r[3], params_2r[0]*0, device=device)
    elif pole_class == 3:
        params_1r1c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_1r1c[0], params_1r1c[0]*0, params_1r1c[1], params_1r1c[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r1c[2], params_1r1c[3], params_1r1c[4], params_1r1c[5], device=device)
    elif pole_class == 4:
        params_2c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_2c[0], params_2c[1], params_2c[2], params_2c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2c[4], params_2c[5], params_2c[6], params_2c[7], device=device)
    elif pole_class == 5:
        params_3r = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_3r[0], params_3r[0]*0, params_3r[1], params_3r[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3r[2], params_3r[0]*0, params_3r[3], params_3r[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3r[4], params_3r[0]*0, params_3r[5], params_3r[0]*0, device=device)
    elif pole_class == 6:
        params_2r1c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_2r1c[0], params_2r1c[0]*0, params_2r1c[1], params_2r1c[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r1c[2], params_2r1c[0]*0, params_2r1c[3], params_2r1c[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_2r1c[4], params_2r1c[5], params_2r1c[6], params_2r1c[7], device=device)
    elif pole_class == 7:
        params_1r2c = pole_params
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, params_1r2c[0], params_1r2c[0]*0, params_1r2c[1], params_1r2c[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r2c[2], params_1r2c[3], params_1r2c[4], params_1r2c[5], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_1r2c[6], params_1r2c[7], params_1r2c[8], params_1r2c[9], device=device)
    elif pole_class == 8:
        params_3c = pole_params
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, params_3c[0], params_3c[1], params_3c[2], params_3c[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3c[4], params_3c[5], params_3c[6], params_3c[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, params_3c[8], params_3c[9], params_3c[10], params_3c[11], device=device)
    return curve_pred







