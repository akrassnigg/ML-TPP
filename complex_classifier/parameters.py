#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Parameters to control the input and behaviour of the code
"""
import pandas as pd
import numpy as np
import os


##############################################################################
##############   General   ###################################################
##############################################################################
# Data directory
data_dir = './data/' 

# Sampling grid
standard_re = pd.read_csv(os.path.join(data_dir, "integration_gridpoints.csv")).to_numpy().reshape(-1)
standard_im = np.linspace(0., 0., num=64)

# Range of real pole positions:
re_max = 0.5
re_min = -10.

# Range of imaginary pole positions (for cc pole pairs only):
im_max = 10.
im_min = 0.0

# Range of absolute values of pole coefficients (for real and imag part of coeff):
coeff_re_max = 1.0
coeff_re_min = 0.0
coeff_im_max = 1.0
coeff_im_min = 0.0

##############################################################################
##############   Classifier   ################################################
##############################################################################
# Directories
dir_classifier        = './data_classifier/'
data_dir_classifier   = dir_classifier + 'data/'
log_dir_classifier    = dir_classifier + 'logs/'
models_dir_classifier = dir_classifier + 'models/'

# Number of data points
n_examples_classifier = 18  # can be a single int or a list of ints, one for each class (which can also be 0 to drop the class).
experimental_num_use_data_classifier = 0# can be a single int or a list of ints, one for each class (which can also be 0 to drop the class). Set to 0 to use all data available

# Properties of drop_small_poles and drop_near_poles
fact_classifier    = np.inf  # set to very large value to not drop any samples
dst_min_classifier = 0.0     # set to 0 to not drop any samples

# Data split
train_portion_classifier = 0.8
val_portion_classifier   = 0.1
test_portion_classifier  = 0.1

# Network and training hyperparameters
### ANN architecture
architecture_classifier = 'FC1'
in_features_classifier  = 133
out_features_classifier = 9
hidden_dim_1_classifier = 32
### Regularization
weight_decay_classifier = 0.0        
### Training hparams
batch_size_classifier    = 32
learning_rate_classifier = 1e-3
epochs_classifier        = int(1e15)

##############################################################################
##############   Regressors   ################################################
##############################################################################
# Class to be learned
class_regressor = 0

# Directories
regressor_subdirs = ['0-1r',
                   '1-1c',
                   '2-2r',
                   '3-1r1c',
                   '4-2c',
                   '5-3r',
                   '6-2r1c',
                   '7-1r2c',
                   '8-3c',]  
dir_regressors       = './data_regressor/'
dir_regressor        = dir_regressors + regressor_subdirs[class_regressor] + '/'
data_dir_regressor   = dir_regressor + 'data/'
log_dir_regressor    = dir_regressor + 'logs/'
models_dir_regressor = dir_regressor + 'models/'

# Number of data points
n_examples_regressor = 2000000

# Data split
train_portion_regressor = 0.98
val_portion_regressor   = 0.01
test_portion_regressor  = 0.01

# Network and training hyperparameters
### ANN Architecture
architecture_regressor = 'FC6'
out_list               = [2,4,4,6,8,6,8,10,12]
out_features_regressor = out_list[class_regressor]  # depends on the pole class
in_features_regressor  = 64
hidden_dim_1_regressor = 128
hidden_dim_2_regressor = hidden_dim_1_regressor
hidden_dim_3_regressor = hidden_dim_2_regressor
hidden_dim_4_regressor = hidden_dim_3_regressor
hidden_dim_5_regressor = hidden_dim_4_regressor
hidden_dim_6_regressor = hidden_dim_5_regressor 
### Regularization
weight_decay_regressor       = 0.0
### Training hparams
batch_size_regressor         = 1000
learning_rate_regressor      = 1e-3   
epochs_regressor             = int(1e15)
val_check_interval_regressor = 100
num_use                      = 50
training_step_regressor      = 0




