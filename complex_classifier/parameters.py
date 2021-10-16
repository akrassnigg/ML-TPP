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
standard_im = np.linspace(0., 0., num=len(standard_re))

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

##############################################
############### Data Creation ################
##############################################
# Number of data points to be created: can be a single int or a list of ints, one for each class (which can also be 0 to drop the class)
n_examples_classifier = 10000

# Properties of drop_small_poles and drop_near_poles
# Shall small poles be dropped; set to very large value to not drop any samples
fact_classifier    = np.inf  
# Shall samples with close poles be dropped; set to 0 to not drop any samples
dst_min_classifier = 0.0     

# Scipy curve_fit parameters
# Fitting method
method_classifier      = ['lm', 'dogbox', 'dogbox', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf', 'trf'] 
# Use parameter boundaries?
with_bounds_classifier = [False, False, True, False, True, True, True, True, True, True, True, True, True, True] 
# Initial guess of parameters
p0_classifier          = ['random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random'] 
# How many times shall we try to fit the data? Note: Values>1 only make sense if p0='random'
num_tries_classifier   = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  
# ~ Maximal number of optimization steps
maxfev_classifier      = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
# Convergence parameter: can be a single int or a list of ints, one for each class (->list of lists)
xtol_classifier        = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]  

##############################################
###############   Training   #################
##############################################
# Number of data points to be used: can be a single int or a list of ints, one for each class (which can also be 0 to drop the class). Set to 0 to use all data available
num_use_data_classifier = 0
# Indices of data_x that shall be used to train the classifier
use_indices_classifier  = np.hstack([   np.arange(69*0, 69*7)  ])
input_name_classifier   = 'lm+dogbox_b=F+dogbox_b=T+trf_b=F+trf_b=Tx3'

# Data split
train_portion_classifier = 0.8
val_portion_classifier   = 0.1
test_portion_classifier  = 0.1

# Network hyperparameters
# ANN architecture
architecture_classifier = 'FC3'
in_features_classifier  = len(use_indices_classifier)
out_features_classifier = 9
hidden_dim_1_classifier = 8
hidden_dim_2_classifier = 8
hidden_dim_3_classifier = 8
hidden_dim_4_classifier = 0
hidden_dim_5_classifier = 0
hidden_dim_6_classifier = 0
 
# Training hyperparameters
optimizer_classifier          = 'Adam'
batch_size_classifier         = 32
learning_rate_classifier      = 1e-3
# Maximal number of epochs
epochs_classifier             = int(1e15)
val_check_interval_classifier = 0.1
# Early Stopping patience
es_patience_classifier        = 20

# Regularization
weight_decay_classifier = 0.0    
drop_prob_1_classifier  = 0.0   
drop_prob_2_classifier  = 0.0   
drop_prob_3_classifier  = 0.0   
drop_prob_4_classifier  = 0.0   
drop_prob_5_classifier  = 0.0   
drop_prob_6_classifier  = 0.0   

# Do multiple runs and average test_acc over them?
num_runs_classifier           = 1

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

##############################################
############### Data Creation ################
##############################################
# Number of data points
n_examples_regressor = 1000000

# Properties of drop_small_poles and drop_near_poles
# Shall small poles be dropped; set to very large value to not drop any samples
fact_regressor    = np.inf  
# Shall samples with close poles be dropped; set to 0 to not drop any samples
dst_min_regressor = 0.0   

##############################################
###############   Training   #################
##############################################
# Number of data points to be used: can be a single int or a list of ints, one for each class (which can also be 0 to drop the class). Set to 0 to use all data available
num_use_data_regressor = 0
# After how many epochs shall the data be updated
num_epochs_use_regressor     = int(1e15)  

# Training mode: 0: start from scratch, 1: resume training from checkpoint
training_step_regressor      = 0 
# Name of the ckpt to resume from (must be inside models folder of the regressor)
name_ckpt_regressor          = 'name.ckpt' 

# Data split
train_portion_regressor = 0.98
val_portion_regressor   = 0.01
test_portion_regressor  = 0.01

# Network hyperparameters
# ANN Architecture
architecture_regressor = 'FC6'
out_list               = [2,4,4,6,8,6,8,10,12]
out_features_regressor = out_list[class_regressor]  # depends on the pole class
in_features_regressor  = len(standard_re)
hidden_dim_1_regressor = 32
hidden_dim_2_regressor = hidden_dim_1_regressor
hidden_dim_3_regressor = hidden_dim_2_regressor
hidden_dim_4_regressor = hidden_dim_3_regressor
hidden_dim_5_regressor = hidden_dim_4_regressor
hidden_dim_6_regressor = hidden_dim_5_regressor 

# Training hyperparameters
optimizer_regressor          = 'Adam'
batch_size_regressor         = 1000
learning_rate_regressor      = 1e-3   
# Maximal number of epochs
epochs_regressor             = int(1e15)
val_check_interval_regressor = 0.1
# Early Stopping patience
es_patience_regressor        = 20

# Regularization
weight_decay_regressor = 0.0
drop_prob_1_regressor  = 0.0   
drop_prob_2_regressor  = 0.0   
drop_prob_3_regressor  = 0.0   
drop_prob_4_regressor  = 0.0   
drop_prob_5_regressor  = 0.0   
drop_prob_6_regressor  = 0.0 

# Do multiple runs and average test_loss over them?
num_runs_regressor           = 1





