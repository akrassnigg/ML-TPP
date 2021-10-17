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
import matplotlib.pyplot as plt

from lib.training_data_generation_regressor import create_training_data_regressor
from lib.architectures import FC6
from lib.standardization_functions import rm_std_data
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

def myL1_norm(y_hat, y, pole_class, std_path,
              re_max, re_min, im_max, im_min, 
              coeff_re_max, coeff_re_min, 
              coeff_im_max, coeff_im_min):
    '''
    Removes standardization from y and y_hat, normalizes their elements to [0,1] and returns the L1 error
    
    y, y_hat: torch.Tensors of equal shape
        Predictions and actual values
        
    pole_class: int: 0-8
        The pole class
        
    std_path: str
        Path to the folder, where variances and means files shall be stored
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box
        
    return: scalar torch.Tensor
        The L1 loss/error
    '''
    
    y     = torch.from_numpy(rm_std_data(data=y.cpu().detach().numpy(),     with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    y_hat = torch.from_numpy(rm_std_data(data=y_hat.cpu().detach().numpy(), with_mean=True, 
                                     std_path=std_path, name_var='variances_params.npy', name_mean='means_params.npy'))
    
    if pole_class == 0:
        max_params = np.array([[re_max],[coeff_re_max]])
        min_params = np.array([[re_min],[-coeff_re_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 1:
        max_params = np.array([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 2:
        max_params = np.array([[re_max],[coeff_re_max]])
        min_params = np.array([[re_min],[-coeff_re_max]])
        min_params = np.transpose(np.tile(min_params, (2,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (2,len(y))),(1,0))
    elif pole_class == 3:
        max_params = np.array([[re_max],[coeff_re_max], [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 4:
        max_params = np.array([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (2,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (2,len(y))),(1,0))
    elif pole_class == 5:
        max_params = np.array([[re_max],[coeff_re_max]])
        min_params = np.array([[re_min],[-coeff_re_max]])
        min_params = np.transpose(np.tile(min_params, (3,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (3,len(y))),(1,0))
    elif pole_class == 6:
        max_params = np.array([[re_max],[coeff_re_max],  [re_max],[coeff_re_max],  [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[-coeff_re_max], [re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 7:
        max_params = np.array([[re_max],[coeff_re_max],  [re_max],[im_max],[coeff_re_max],[coeff_im_max],   [re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[-coeff_re_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max], [re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (1,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (1,len(y))),(1,0))
    elif pole_class == 8:
        max_params = np.array([[re_max],[im_max],[coeff_re_max],[coeff_im_max]])
        min_params = np.array([[re_min],[im_min],[-coeff_re_max],[-coeff_im_max]])
        min_params = np.transpose(np.tile(min_params, (3,len(y))),(1,0))
        max_params = np.transpose(np.tile(max_params, (3,len(y))),(1,0))
    else:
        sys.exit("Undefined label.")
    
    y     = (y - min_params) / (max_params - min_params)
    y_hat = (y_hat - min_params) / (max_params - min_params)
    
    return F.l1_loss(y_hat, y)


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
        
    return: scalar torch.Tensor
        The MSE Reconstruction Loss
    '''
    def __init__(self, pole_class, std_path, grid_x):
        super(pole_reconstruction_loss,self).__init__()
        self.pole_class = pole_class
        self.std_path   = std_path
        self.grid_x     = grid_x
        self.prepared   = False
        
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
        loss   = F.mse_loss(x_pred, x)
        ###########################################################
        #f1 = x_pred[0,:].detach().cpu().numpy()
        #f2 = x[0,:].detach().cpu().numpy()
        #plt.plot(f1)
        #plt.plot(f2)
        #plt.show()
        #print(loss)
        ###########################################################
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
        self.Reconstruction_loss = pole_reconstruction_loss(pole_class=pole_class, std_path=std_path, grid_x=grid_x)
 
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)
               
        parameter_loss = F.mse_loss(y_hat, y)
        self.log('train_parameter_loss', parameter_loss, on_step=True, on_epoch=False) 

        reconstruction_loss = self.Reconstruction_loss(y_hat, x)  
        self.log('train_reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=False) 

        # Loss to be used for training:
        train_loss = parameter_loss + 1e-100*reconstruction_loss
        self.log('train_loss', train_loss, on_step=True, on_epoch=False) 

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
           
        parameter_loss = F.mse_loss(y_hat, y)
        self.log('val_parameter_loss', parameter_loss, on_step=False, on_epoch=True) 

        reconstruction_loss = self.Reconstruction_loss(y_hat, x)  
        self.log('val_reconstruction_loss', reconstruction_loss, on_step=False, on_epoch=True) 
         
        parameter_error_percent = myL1_norm(y_hat, y, pole_class=self.hparams.pole_class, std_path=self.hparams.std_path,
                                     re_max=self.hparams.re_max, re_min=self.hparams.re_min, 
                                     im_max=self.hparams.im_max, im_min=self.hparams.im_min, 
                                     coeff_re_max=self.hparams.coeff_re_max, coeff_re_min=self.hparams.coeff_re_min, 
                                     coeff_im_max=self.hparams.coeff_im_max, coeff_im_min=self.hparams.coeff_im_min)
        self.log('val_parameter_error', parameter_error_percent, on_step=False, on_epoch=True)
        
        # Loss to be used for validation:
        val_loss = parameter_loss + 1e-100*reconstruction_loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True) 

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        parameter_loss = F.mse_loss(y_hat, y)
        self.log('test_parameter_loss', parameter_loss, on_step=False, on_epoch=True) 

        reconstruction_loss = self.Reconstruction_loss(y_hat, x)  
        self.log('test_reconstruction_loss', reconstruction_loss, on_step=False, on_epoch=True) 
        
        parameter_error_percent = myL1_norm(y_hat, y, pole_class=self.hparams.pole_class, std_path=self.hparams.std_path,
                                     re_max=self.hparams.re_max, re_min=self.hparams.re_min, 
                                     im_max=self.hparams.im_max, im_min=self.hparams.im_min, 
                                     coeff_re_max=self.hparams.coeff_re_max, coeff_re_min=self.hparams.coeff_re_min, 
                                     coeff_im_max=self.hparams.coeff_im_max, coeff_im_min=self.hparams.coeff_im_min)
        self.log('test_parameter_error', parameter_error_percent, on_step=False, on_epoch=True)

        # Loss to be used for testing:
        test_loss = parameter_loss + 1e-100*reconstruction_loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True) 

        return test_loss

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









#