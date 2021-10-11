# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Functions needed to apply the finished classifier to real data

"""
import numpy as np
import os
import torch
import scipy

from lib.standardization_functions import std_data
from lib.pole_classifier import Pole_Classifier
from lib.training_data_generation_classifier import get_data_x

def get_classifier_preds(grid_x, data_y, 
                         re_max, re_min, im_max, im_min, 
                         coeff_re_max, coeff_re_min, 
                         coeff_im_max, coeff_im_min,
                         do_std, model_path, 
                         with_bounds=True, p0='default', method='trf', 
                         maxfev=100000, num_tries=1, xtol = 1e-8):
    '''
    Get predictions from trained Pole Classifier(s) 
    
    grid_x: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Gridpoints
    
    data_y: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Function Values
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    do_std: bool
        Shall the data be standardized to the standardization, that was used to train the Classifier?
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Classifier.
            
            If do_std=True, there must additionally be a subdirectory called 'data' containing the standardization files called "variances.npy" and "means.npy".
    
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
    
    returns: int, int, ndarray of shape (l,), where l is the number of checkpoints in the "models" subdirectory
        Class Predictions: Hard Averaging, Soft Averaging, Array with Predictions from the individual Checkpoints
    '''
    # Data Preparation
    # Get data_x
    data_x = get_data_x(data_y=data_y, grid_x=grid_x, 
                        re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                        coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                        coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                        with_bounds=with_bounds, p0=p0,
                        method=method, maxfev=maxfev, num_tries=num_tries, xtol=xtol) 
    # Apply standardization
    if do_std:
        data_x = std_data(data=data_x, std_path=os.path.join(model_path, 'data/'), with_mean=True)
    
    # cut away out_re, which was not used to train the classifier
    data_x = data_x[:,0:69]
    
    # Get predictions from Classifiers
    class_pred = []
    for filename in os.listdir(os.path.join(model_path, 'models/')):
        model = Pole_Classifier.load_from_checkpoint(os.path.join(model_path, 'models/', filename))
        model.eval()
        pred = model(torch.from_numpy(data_x.astype('float32')))
        class_pred.append( pred.detach().numpy() )
        del model
    class_pred = np.array(class_pred)
    class_pred = np.swapaxes(class_pred,0,1)
    
    # Step 3a: Hard Averaging - Take most frequent prediction (=mode)
    class_pred_hard = np.argmax(class_pred, axis=2)
    class_pred_hard
    class_pred_hard = scipy.stats.mode(class_pred_hard, axis=1)[0] 
    class_pred_hard = torch.from_numpy(class_pred_hard)
    
    # Step 3b: Soft Averaging - sum over probabilities and the take argmax
    class_pred_soft = np.sum(class_pred, axis=1)
    class_pred_soft = np.argmax(class_pred_soft, axis=1).reshape((-1,1))
    class_pred_soft = torch.from_numpy(class_pred_soft)
    
    return class_pred_soft.item(), class_pred_hard.item(), np.argmax(class_pred, axis=2).reshape(-1)

















