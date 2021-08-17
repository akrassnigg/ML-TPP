#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

from parameters import class_regressor, data_dir_regressor
from parameters import re_max, re_min, im_max, im_min, coeff_max
from parameters import num_use
from lib.training_data_generation_regressor import create_training_data_regressor
from lib.architectures import FC6
from lib.standardization_functions import rm_std_data


##############################################################################
###################   Data-related Classes   #################################
##############################################################################
class PoleDataSet_Regressor(Dataset):
    def __init__(self, pole_class, data_x, data_dir):
        self.count      = 0  # counts the number of times __getitem__ was called
        self.pole_class = pole_class
        self.data_dir   = data_dir
        self.data_x     = data_x

        self.data_X = np.load(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), allow_pickle=True).astype('float32')
        self.data_Y = np.load(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), allow_pickle=True).astype('float32')
            
        print("Checking shape of loaded data: X: ", np.shape(self.data_X))
        print("Checking shape of loaded data: y: ", np.shape(self.data_Y))
        
    def update_dataset(self):
        num_data = len(self)
        
        data_x, data_y = create_training_data_regressor(length=num_data, pole_class=self.pole_class, mode='update', 
                         data_x=self.data_x, data_dir=self.data_dir)
            
        self.data_X = np.array(data_x, dtype='float32')
        self.data_Y = np.array(data_y, dtype='float32')
          
        print("Checking shape of updated data: X: ", np.shape(self.data_X))
        print("Checking shape of updated data: y: ", np.shape(self.data_Y))
        
    def __len__(self):
        num_of_data_points = len(self.data_X)
        return num_of_data_points

    def __getitem__(self, idx):
        
        if self.count == num_use*len(self):
            self.update_dataset()
            self.count = 0
        
        self.count += 1
        return self.data_X[idx], self.data_Y[idx]
     
        
class PoleDataModule_Regressor(pl.LightningDataModule):
    def __init__(self, pole_class, data_x, data_dir: str, batch_size: int, train_portion: float, validation_portion: float, test_portion: float):
        super().__init__()
        self.pole_class         = pole_class
        self.data_x             = data_x
        self.data_dir           = data_dir
        self.batch_size         = batch_size
        self.train_portion      = train_portion
        self.validation_portion = validation_portion
        self.test_portion       = test_portion

    def setup(self, stage):
        all_data = PoleDataSet_Regressor(pole_class=self.pole_class, data_x=self.data_x, data_dir=self.data_dir)
        
        num_data = len(all_data)
        print("Length of all_data: ", num_data)
        
        self.training_number = int(self.train_portion*num_data)
        self.validation_number = int(self.validation_portion*num_data)
        self.test_number = int(num_data - self.training_number - self.validation_number)
        
        print("Data splits: ", self.training_number, self.validation_number, self.test_number, num_data)
        train_part, val_part, test_part = random_split(all_data, [self.training_number, self.validation_number, self.test_number]
                                                       , generator=torch.Generator().manual_seed(1234))

        self.train_dataset = train_part
        self.val_dataset = val_part
        self.test_dataset = test_part

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,   batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=self.batch_size)

    
##############################################################################
###########################   Regressor    ###################################
##############################################################################


def myL1(y_hat, y, std_path):
    '''
    Removes standardization from y and y_hat and returns the L1 error
    
    y, y_hat: torch.Tensors of equal shape
        Predictions and actual values
        
    std_path: str
        Path to the folder, where variances and means files shall be stored
        
    return: scalar torch.Tensor
        The L1 loss/error
    '''
    
    y     = torch.from_numpy(rm_std_data(data=y.cpu().detach().numpy(),     with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    y_hat = torch.from_numpy(rm_std_data(data=y_hat.cpu().detach().numpy(), with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    return F.l1_loss(y_hat, y)

def myL1_norm(y_hat, y, pole_class, std_path):
    '''
    Removes standardization from y and y_hat, normalizes their elements to [0,1] and returns the L1 error
    
    y, y_hat: torch.Tensors of equal shape
        Predictions and actual values
        
    pole_class: int: 0-8
        The pole class
        
    std_path: str
        Path to the folder, where variances and means files shall be stored
        
    return: scalar torch.Tensor
        The L1 loss/error
    '''
    
    y     = torch.from_numpy(rm_std_data(data=y.cpu().detach().numpy(),     with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    y_hat = torch.from_numpy(rm_std_data(data=y_hat.cpu().detach().numpy(), with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    
    if pole_class == 0:
        max_params = np.array([[re_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 1:
        max_params = np.array([[re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 2:
        max_params = np.array([[re_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (2,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (2,len(y))),(1,0))
    elif pole_class == 3:
        max_params = np.array([[re_max],[coeff_max], [re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max], [re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 4:
        max_params = np.array([[re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (2,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (2,len(y))),(1,0))
    elif pole_class == 5:
        max_params = np.array([[re_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (3,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (3,len(y))),(1,0))
    elif pole_class == 6:
        max_params = np.array([[re_max],[coeff_max],  [re_max],[coeff_max],  [re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max], [re_min],[-coeff_max], [re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 7:
        max_params = np.array([[re_max],[coeff_max],  [re_max],[im_max],[coeff_max],[coeff_max],   [re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[-coeff_max], [re_min],[im_min],[-coeff_max],[-coeff_max], [re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 8:
        max_params = np.array([[re_max],[im_max],[coeff_max],[coeff_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_max],[-coeff_max]])
        min_params = np.transpose(np.tile(min_params, (3,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (3,len(y))),(1,0))
    else:
        sys.exit("Undefined label.")
    
    y     = (y - min_params) / (max_params - min_params)
    y_hat = (y_hat - min_params) / (max_params - min_params)
    
    return F.l1_loss(y_hat, y)

    
class Pole_Regressor(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the parameters of a complex structure in the vector
    """
    def __init__(self, 
                 # Regularization
                 weight_decay:  float = 0.0,
                 
                 # Training-related hyperparameters
                 learning_rate: float = 1e-3, 
                 
                 # ANN Architecture
                 architecture: str = 'FC6', 
                 
                 #Additional arguments needed for initialization of the ANN Architecture
                 **kwargs
                 ):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = globals()[architecture](**kwargs)
 
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)
               
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
           
        val_loss = F.l1_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        
        val_norm_l1_loss = myL1_norm(y_hat, y, pole_class=class_regressor, std_path=data_dir_regressor)
        self.log('val_norm_l1_loss', val_norm_l1_loss, on_step=False, on_epoch=True)
        
        val_l1_loss = myL1(y_hat, y, std_path=data_dir_regressor)
        self.log('val_l1_loss', val_l1_loss, on_step=False, on_epoch=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        test_loss = F.l1_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}









#