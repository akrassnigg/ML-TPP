#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import matplotlib.pyplot as plt

from lib.training_data_generation_regressor import create_training_data_regressor
from lib.architectures import FC6
from lib.standardization_functions import rm_std_data, rm_std_data_torch
from lib.curve_calc_functions_torch import pole_curve_calc2_torch


##############################################################################
###################   Data-related Classes   #################################
##############################################################################
class PoleDataSet_Regressor(Dataset):
    """
    DataSet of the Pole Regressor
    
    pole_class: int
        The class to be trained
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    data_dir: str
        Directory containing data files
    
    num_use_data: int
        How many of the available datapoints shall be used?
    
    num_epochs_use: int
        After how many training epochs shall the data be refreshed?
    
    fact: numeric>=1
        Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
        
    dst_min: numeric>=0
        Drops parameter configureations, that contain poles, whose positions are nearer to each other than dst_min (complex, euclidean norm)
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: float
        The parameter boundaries
    """
    def __init__(self, pole_class, grid_x, data_dir, num_use_data, num_epochs_use, fact, dst_min,
                 re_max, re_min, im_max, im_min, 
                 coeff_re_max, coeff_re_min, 
                 coeff_im_max, coeff_im_min):
        self.count      = 0  # counts the number of times __getitem__ was called
        self.pole_class = pole_class
        self.data_dir   = data_dir
        self.grid_x     = grid_x
        self.num_epochs_use = num_epochs_use
        self.fact       = fact
        self.dst_min    = dst_min
        self.re_max     = re_max
        self.re_min     = re_min
        self.im_max     = im_max
        self.im_min     = im_min
        self.coeff_re_max = coeff_re_max
        self.coeff_re_min = coeff_re_min
        self.coeff_im_max = coeff_im_max
        self.coeff_im_min = coeff_im_min

        self.data_X = np.load(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), allow_pickle=True).astype('float32')
        self.data_Y = np.load(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), allow_pickle=True).astype('float32')
            
        print("Checking shape of loaded data: X: ", np.shape(self.data_X))
        print("Checking shape of loaded data: y: ", np.shape(self.data_Y))
        
        if num_use_data ==0:   #use all data available
            None
        else:
            #seed_afterward = np.random.randint(low=0, high=1e3)
            #np.random.seed(1234)
            indices = np.arange(len(self.data_Y))
            np.random.shuffle(indices)
            indices = indices[0:num_use_data]
            #np.random.seed(seed_afterward)
            self.data_X = self.data_X[indices]
            self.data_Y = self.data_Y[indices]
            print('Successfully selected a Subset of the Data...')
            print("Checking shape of data: X Subset: ", np.shape(self.data_X))
            print("Checking shape of data: y Subset: ", np.shape(self.data_Y))
        
    def update_dataset(self):
        num_data = len(self)
        
        data_x, data_y = create_training_data_regressor(mode='update', length=num_data, pole_class=self.pole_class, 
                         grid_x=self.grid_x, data_dir=self.data_dir, fact=self.fact, dst_min=self.dst_min,
                         re_max=self.re_max, re_min=self.re_min, im_max=self.im_max, im_min=self.im_min, 
                         coeff_re_max=self.coeff_re_max, coeff_re_min=self.coeff_re_min, 
                         coeff_im_max=self.coeff_im_max, coeff_im_min=self.coeff_im_min)
            
        self.data_X = np.array(data_x, dtype='float32')
        self.data_Y = np.array(data_y, dtype='float32')
          
        print("Checking shape of updated data: X: ", np.shape(self.data_X))
        print("Checking shape of updated data: y: ", np.shape(self.data_Y))
        
    def __len__(self):
        num_of_data_points = len(self.data_X)
        return num_of_data_points

    def __getitem__(self, idx):
        
        if self.count == self.num_epochs_use*len(self):
            self.update_dataset()
            self.count = 0
        
        self.count += 1
        return self.data_X[idx], self.data_Y[idx]
     
        
class PoleDataModule_Regressor(pl.LightningDataModule):
    """
    DataModule of the Pole Regressor
    
    pole_class: int
        The class to be trained
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    data_dir: str
        Directory containing data files
    
    batch_size: int
    
    train_portion, validation_portion, test_portion: float
    
    num_use_data: int
        How many of the available datapoints shall be used?
    
    num_epochs_use: int
        After how many training epochs shall the data be refreshed?
    
    fact: numeric>=1
        Drops parameter configureations, that contain poles, whose out_re is a factor fact smaller, than out_re of the other poles in the sample
        
    dst_min: numeric>=0
        Drops parameter configureations, that contain poles, whose positions are nearer to each other than dst_min (complex, euclidean norm)
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: float
        The parameter boundaries
    """
    def __init__(self, pole_class, grid_x, data_dir: str, batch_size: int, train_portion: float, validation_portion: float, test_portion: float,
                 num_use_data, num_epochs_use, fact, dst_min,
                 re_max, re_min, im_max, im_min, 
                 coeff_re_max, coeff_re_min, 
                 coeff_im_max, coeff_im_min):
        super().__init__()
        self.pole_class         = pole_class
        self.grid_x             = grid_x
        self.data_dir           = data_dir
        self.batch_size         = batch_size
        self.train_portion      = train_portion
        self.validation_portion = validation_portion
        self.test_portion       = test_portion
        self.num_use_data   = num_use_data
        self.num_epochs_use = num_epochs_use
        self.fact           = fact
        self.dst_min        = dst_min
        self.re_max     = re_max
        self.re_min     = re_min
        self.im_max     = im_max
        self.im_min     = im_min
        self.coeff_re_max = coeff_re_max
        self.coeff_re_min = coeff_re_min
        self.coeff_im_max = coeff_im_max
        self.coeff_im_min = coeff_im_min

    def setup(self, stage):
        all_data = PoleDataSet_Regressor(pole_class=self.pole_class, grid_x=self.grid_x, data_dir=self.data_dir, 
                                         num_use_data=self.num_use_data, num_epochs_use = self.num_epochs_use,
                                         fact=self.fact, dst_min=self.dst_min,
                                         re_max=self.re_max, re_min=self.re_min, im_max=self.im_max, im_min=self.im_min, 
                                         coeff_re_max=self.coeff_re_max, coeff_re_min=self.coeff_re_min, 
                                         coeff_im_max=self.coeff_im_max, coeff_im_min=self.coeff_im_min)
        
        num_data = len(all_data)
        print("Length of all_data: ", num_data)
        
        self.training_number = int(self.train_portion*num_data)
        self.validation_number = int(self.validation_portion*num_data)
        self.test_number = int(num_data - self.training_number - self.validation_number)
        
        print("Data splits: ", self.training_number, self.validation_number, self.test_number, num_data)
        train_part, val_part, test_part = random_split(all_data, [self.training_number, self.validation_number, self.test_number]
                                                       )#, generator=torch.Generator().manual_seed(1234))

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

class pole_reconstruction_loss(torch.nn.Module):
    '''
    Reconstruction Loss for Pole Regressor
    
    y_hat: torch.Tensor of shape (batch_size, out_features_regressor)
        Parameter predictions 
        
    x: torch.Tensor of shape (batch_size, in_features_regressor)
        Actual pole curves 
        
    pole_class: int: 0-8
        The pole class
        
    std_path: str
        Path to the folder, where variances and means files shall be stored
        
    grid_x: np.ndarray or torch.Tensor of shape (in_features_regressor,) or (1,in_features_regressor)
        The integration grid
        
    loss_type: str: 'mse' or 'l1'
        
    return: scalar torch.Tensor
        The MSE Reconstruction Loss
    '''
    def __init__(self, pole_class, std_path, grid_x, loss_type):
        super(pole_reconstruction_loss,self).__init__()
        self.pole_class = pole_class
        self.std_path   = std_path
        self.grid_x     = grid_x
        self.prepared   = False
        self.loss_type  = loss_type
        
    def preparation(self, y_hat):
        # Get device of Tensors 
        self.used_device = y_hat.device
        
        # Prepare grid_x
        if type(self.grid_x) == np.ndarray:
            self.grid_x = torch.from_numpy(self.grid_x)
        self.grid_x = self.grid_x.to(device=self.used_device).float()
        
        # Prepare removal of std from x
        variances_x   = torch.from_numpy(np.load(os.path.join(self.std_path, 'variances.npy'), allow_pickle=True).astype('float32'))
        self.scales_x = (torch.sqrt(variances_x)).to(device=self.used_device)
        
        # Prepare removal of std from y_hat
        variances_y   = torch.from_numpy(np.load(os.path.join(self.std_path, 'variances_params.npy'), allow_pickle=True).astype('float32'))
        self.scales_y = (torch.sqrt(variances_y)).to(device=self.used_device)
        means_y       = torch.from_numpy(np.load(os.path.join(self.std_path, 'means_params.npy'), allow_pickle=True).astype('float32'))
        self.means_y  = means_y.to(device=self.used_device)
        
        # Preparation finished
        self.prepared = True
        
    def forward(self,y_hat, x):
        if self.prepared == False:
            self.preparation(y_hat)
        
        # Remove std from x   
        x      = x * torch.tile(self.scales_x, (x.shape[0],1))
        
        # Remove std from y_hat
        y_hat  = y_hat * torch.tile(self.scales_y, (y_hat.shape[0],1)) + torch.tile(self.means_y, (y_hat.shape[0],1))
        
        # Calculate Pole curves from y_hat
        x_pred = pole_curve_calc2_torch(pole_class=self.pole_class, pole_params=y_hat, grid_x=self.grid_x, device=self.used_device)
        
        # Calculate MSE
        if self.loss_type == 'mse':
            loss   = F.mse_loss(x_pred, x)
        elif self.loss_type == 'l1':
            loss   = F.l1_loss(x_pred, x)
        return loss


class Pole_Regressor(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the parameters of a complex structure in the vector
    
    pole_class: int
        The class that shall be trained
        
    std_path: str
        Folder containing the standardization info of the data (needed for special loss)
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: floats
        The parameter boundaries
    
    grid_x: np.ndarray or torch.Tensor of shape (in_features_regressor,) or (1,in_features_regressor)
        The integration grid
    
    weight_decay, learning_rate: floats
    
    architecture, optimizer: strings
    
    additional kwargs are handed to the architecture class
    """
    def __init__(self, 
                 # The pole class
                 pole_class: int,
                 
                 # Path to the folder, where variances and means files shall be stored
                 std_path: str,
                 
                 # Parameter boundaries
                 re_max, re_min, im_max, im_min, 
                 coeff_re_max, coeff_re_min, 
                 coeff_im_max, coeff_im_min,
                 
                 # The Integration grid
                 grid_x,
                 
                 # Specify the loss
                 parameter_loss_type, reconstruction_loss_type,
                 parameter_loss_coeff, reconstruction_loss_coeff,
                 
                 # Regularization
                 weight_decay:  float = 0.0,
                 
                 # Training-related hyperparameters
                 learning_rate: float = 1e-3, 
                 
                 # ANN Architecture
                 architecture: str = 'FC6', 
                 
                 # Optimizer
                 optimizer: str = 'Adam',
                 
                 #Additional arguments needed for initialization of the ANN Architecture
                 **kwargs
                 ):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = globals()[architecture](**kwargs)
        self.Reconstruction_loss_mse = pole_reconstruction_loss(pole_class=pole_class, std_path=std_path, grid_x=grid_x, loss_type='mse')
        self.Reconstruction_loss_l1  = pole_reconstruction_loss(pole_class=pole_class, std_path=std_path, grid_x=grid_x, loss_type='l1')
        self.Parameter_loss_mse      = nn.MSELoss()
        self.Parameter_loss_l1       = nn.L1Loss()
        
        self.boundary_setup_finished = False
        
    def losses(self, x, y, y_hat):
        parameter_loss_type            = self.hparams.parameter_loss_type
        reconstruction_loss_type       = self.hparams.reconstruction_loss_type
        parameter_loss_coeff           = self.hparams.parameter_loss_coeff
        reconstruction_loss_coeff      = self.hparams.reconstruction_loss_coeff
        
        if parameter_loss_type        == 'mse':
            parameter_loss             = self.Parameter_loss_mse(y_hat, y)
        elif parameter_loss_type      == 'l1':
            parameter_loss             = self.Parameter_loss_l1(y_hat, y)
        if reconstruction_loss_type   == 'mse':
            reconstruction_loss        = self.Reconstruction_loss_mse(y_hat, x)  
        elif reconstruction_loss_type == 'l1':
            reconstruction_loss        = self.Reconstruction_loss_l1(y_hat, x)  

        # loss that is used to train the network:
        loss                           = (parameter_loss_coeff*parameter_loss + 
                                          reconstruction_loss_coeff*reconstruction_loss)
        return [parameter_loss, reconstruction_loss, loss]
 
    def forward(self, x):
        x = self.model(x)
        x = self.apply_boundaries(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)             
        _, _, loss = self.losses(x=x,y=y,y_hat=y_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=False) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)     
        _, _, loss = self.losses(x=x,y=y,y_hat=y_hat)
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, _, loss = self.losses(x=x,y=y,y_hat=y_hat)
        self.log('test_loss', loss, on_step=False, on_epoch=True) 
        
        #######################################################################
        #######################################################################
        # Log Test losses without std (remove std)
        # Remove std from x, y and y_hat
        x           = rm_std_data_torch(data=x, with_mean=False, 
                                        std_path=self.hparams.std_path, name_var="variances.npy")
        y           = rm_std_data_torch(data=y, with_mean=True, 
                                        std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy')
        y_hat       = rm_std_data_torch(data=y_hat, with_mean=True, 
                                        std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy')
        
        # Calculate predicted curve
        grid        = torch.from_numpy(self.hparams.grid_x)
        x_hat       = pole_curve_calc2_torch(pole_class=self.hparams.pole_class, pole_params=y_hat, grid_x=grid, device=x.device)
        
        # Parameters MSE 
        params_mse = F.mse_loss(y_hat, y) 
        self.log('test_params_mse_nostd', params_mse, on_step=False, on_epoch=True) 
        
        # Parameter_i MSE 
        for i in range(y.shape[1]):
            params_i_mse = F.mse_loss(y_hat[:,i], y[:,i]) 
            self.log('test_params_{}_mse_nostd'.format(i), params_i_mse, on_step=False, on_epoch=True)
        
        # Parameters MAE
        params_mae = F.l1_loss(y_hat, y) 
        self.log('test_params_mae_nostd', params_mae, on_step=False, on_epoch=True)
        
        # Parameter_i MAE
        for i in range(y.shape[1]):
            params_i_mae = F.l1_loss(y_hat[:,i], y[:,i]) 
            self.log('test_params_{}_mae_nostd'.format(i), params_i_mae, on_step=False, on_epoch=True)
        
        # Reconstruction MSE
        reconstruction_mse = F.mse_loss(x_hat, x) 
        self.log('reconstruction_mse_nostd', reconstruction_mse, on_step=False, on_epoch=True)
        
        # Reconstruction MAE
        reconstruction_mae = F.l1_loss(x_hat, x) 
        self.log('reconstruction_mae_nostd', reconstruction_mae, on_step=False, on_epoch=True)
        #######################################################################
        #######################################################################

        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}
    
    def apply_boundaries(self, x):
        if self.boundary_setup_finished == False:
            # Get lower and upper bounds: 
            re_max = self.hparams.re_max
            re_min = float('-inf')
            im_max = float('inf')
            im_min = float('-inf')
            coeff_re_max = float('inf')
            coeff_im_max = float('inf')
            if self.hparams.pole_class == 0:
                max_params = torch.Tensor([[re_max],[coeff_re_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max]])
                min_params = torch.transpose(torch.tile(min_params, (1,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (1,len(x))),1,0)
            elif self.hparams.pole_class == 1:
                max_params = torch.Tensor([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (1,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (1,len(x))),1,0)
            elif self.hparams.pole_class == 2:
                max_params = torch.Tensor([[re_max],[coeff_re_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max]])
                min_params = torch.transpose(torch.tile(min_params, (2,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (2,len(x))),1,0)
            elif self.hparams.pole_class == 3:
                max_params = torch.Tensor([[re_max],[coeff_re_max], [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (1,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (1,len(x))),1,0)
            elif self.hparams.pole_class == 4:
                max_params = torch.Tensor([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (2,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (2,len(x))),1,0)
            elif self.hparams.pole_class == 5:
                max_params = torch.Tensor([[re_max],[coeff_re_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max]])
                min_params = torch.transpose(torch.tile(min_params, (3,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (3,len(x))),1,0)
            elif self.hparams.pole_class == 6:
                max_params = torch.Tensor([[re_max],[coeff_re_max],  [re_max],[coeff_re_max],  [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max], [re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (1,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (1,len(x))),1,0)
            elif self.hparams.pole_class == 7:
                max_params = torch.Tensor([[re_max],[coeff_re_max],  [re_max],[im_max],[coeff_re_max],[coeff_im_max],   [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (1,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (1,len(x))),1,0)
            elif self.hparams.pole_class == 8:
                max_params = torch.Tensor([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
                min_params = torch.Tensor([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
                min_params = torch.transpose(torch.tile(min_params, (3,len(x))),1,0)
                max_params = torch.transpose(torch.tile(max_params, (3,len(x))),1,0)
            else:
                sys.exit("Undefined label.")    
                
            #Apply the standardization to the boundaries
            variances_y = torch.from_numpy(np.load(os.path.join(self.hparams.std_path, 'variances_params.npy'), allow_pickle=True).astype('float32'))
            scales_y    = (torch.sqrt(variances_y))
            means_y     = torch.from_numpy(np.load(os.path.join(self.hparams.std_path, 'means_params.npy'), allow_pickle=True).astype('float32'))
            min_params  = (min_params - torch.tile(means_y, (min_params.shape[0],1))) / torch.tile(scales_y, (min_params.shape[0],1))
            max_params  = (max_params - torch.tile(means_y, (max_params.shape[0],1))) / torch.tile(scales_y, (max_params.shape[0],1))
            
            device      = x.device
            self.min_params = min_params.to(device=device)
            self.max_params = max_params.to(device=device)
            self.boundary_setup_finished = True  
            
        # Apply boundaries
        x = torch.maximum(x, self.min_params)
        x = torch.minimum(x, self.max_params)
        return x




#