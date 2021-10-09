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
from lib.architectures import FC1
from lib.pole_classifier import Pole_Classifier
from lib.training_data_generation_classifier import get_data_x

def get_classifier_preds(grid_x, data_y, with_bounds, do_std, model_path):
    '''
    Get predictions from trained Pole Classifier(s) 
    
    grid_x: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Gridpoints
    
    data_y: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Function Values
    
    with_bounds: bool
        Shall Scipy curve_fit be restricted to boundaries given by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max? (see also: prepare_data9)
    
    do_std: bool
        Shall the data be standardized to the standardization, that was used to train the Classifier?
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Classifier.
            
            If do_std=True, there must additionally be a subdirectory called 'data' containing the standardization files called "variances.npy" and "means.npy".
    
    returns: int, int, ndarray of shape (l,), where l is the number of checkpoints in the "models" subdirectory
        Class Predictions: Hard Averaging, Soft Averaging, Array with Predictions from the individual Checkpoints
    '''
    # Data Preparation
    # Get data_x
    data_x = get_data_x(out_re=data_y, grid_x=grid_x, with_bounds=with_bounds) 
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

















