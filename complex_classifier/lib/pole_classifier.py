#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template

"""
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import WeightedRandomSampler

from lib.architectures import FC1


##############################################################################
###################   Data-related Classes   #################################
##############################################################################

class PoleDataSet_Classifier(Dataset):
    def __init__(self, data_dir):
        self.data_X = np.load(os.path.join(data_dir, "various_poles_data_classifier_x.npy"), allow_pickle=True).astype('float32')
        self.data_Y = np.load(os.path.join(data_dir, "various_poles_data_classifier_y.npy"), allow_pickle=True).astype('int64').reshape((-1,1))
        
        print("Checking shape of loaded data: X: ", np.shape(self.data_X))
        print("Checking shape of loaded data: y: ", np.shape(self.data_Y))

    def __len__(self):
        num_of_data_points = len(self.data_X)
        print("Successfully loaded pole training data of length: ", num_of_data_points)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]


class PoleDataModule_Classifier(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, train_portion: float, validation_portion: float, test_portion: float):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.test_portion = test_portion

    def setup(self, stage):
        all_data = PoleDataSet_Classifier(self.data_dir)
        
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
        
        self.train_weights = self.get_train_weights()
        print('Label weights:')
        print(self.label_weights)
        
    def get_train_weights(self):
        #get train weights for weighted random sampler
        train_labels = self.train_dataset[:][1]
        max_label    = np.max(train_labels)
        
        label_counter = []
        for i in range(max_label+1):
            label_counter.append((train_labels == i).sum())
        
        label_counter = np.array(label_counter)
        self.label_weights = 1/label_counter
        
        train_weights = train_labels.astype('float32')
        for i in range(max_label+1):
            train_weights[train_weights==i] = self.label_weights[i]
        
        train_weights = train_weights.reshape(-1)
        return train_weights

    def train_dataloader(self):
        sampler = WeightedRandomSampler(weights=self.train_weights, num_samples=len(self.train_weights))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    
##############################################################################
###########################   Classifier   ###################################
##############################################################################


class Pole_Classifier(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the class of a complex structure in the vector
    """
    def __init__(self, 
                 # Regularization
                 weight_decay:  float = 0.0,
                 
                 # Training-related hyperparameters
                 learning_rate: float = 1e-3, 
                 
                 # ANN Architecture
                 architecture: str = 'FC1', 
                 
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
        y = y.view(-1)
        
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1)
        
        val_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)               
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1)
        
        test_loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}

    


#
