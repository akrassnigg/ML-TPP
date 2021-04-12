#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@author: andreaskrassnigg

Classifier based on pytorch basic template
"""


import os
import sys

import numpy as np

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning.metrics.utils as ptlmu
from pytorch_lightning.utilities.seed import seed_everything

import optuna
from optuna.trial import TrialState
from optuna.integration import PyTorchLightningPruningCallback

do_pruning = True

EPOCHS = 30
TIMEOUT = 3600
NTRIALS = 1000
VAL_PORTION = 0.2
TEST_PORTION = 0.2
NUM_WORKERS = 0
CLASSES = 2

basedir = './data'
logdir = "logs"

model_timestamp = int(time.time())

tb_logger = TensorBoardLogger(logdir, name='ml-tpp-classifier', version=model_timestamp)

class RandomClasses(Dataset):
    def __init__(self, data_dir):

        self.data_X = np.load(data_dir+"/random_test_data_classifier_x.npy", allow_pickle=True).astype('float32')
        self.data_y = np.load(data_dir+"/random_test_data_classifier_y.npy", allow_pickle=True).astype('int')

    def __len__(self):
        num_of_data_points = len(self.data_X)
        print("Successfully loaded random training data of length: ", num_of_data_points)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]
    



class PoleClasses(Dataset):
    def __init__(self, data_dir):

        self.data_X = np.load(data_dir+"/various_poles_data_classifier_x.npy", allow_pickle=True).astype('float32')
        self.data_y = np.reshape(np.load(data_dir+"/various_poles_data_classifier_y.npy", allow_pickle=True).astype('int'), (-1,1))
        
        print("Checking shape of loaded data: X: ", np.shape(self.data_X))
        print("Checking shape of loaded data: y: ", np.shape(self.data_y))

    def __len__(self):
        num_of_data_points = len(self.data_X)
        print("Successfully loaded pole training data of length: ", num_of_data_points)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]
    
    
net_input_dim = len(PoleClasses(basedir)[0][0])



class DeepNet1(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    a vector of outputs, i.e. vector to vector
    """

    def __init__(self,
                 in_features: int = 20,
                 hidden_dim: int = 1000,
                 out_features: int = 4,
                 drop_prob: float = 0.2,
                 learning_rate: float = 0.001 * 8,
                 batch_size: int = 20,
                 val_batch_size: int = 2,
                 test_batch_size: int = 2,
                 validation_portion: float = .2,
                 test_portion: float = .2,
                 data_root: str = './data',
                 num_data: int = 200,
                 num_workers: int = 4,
                 **kwargs
                 ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        self.fc_1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.fc_1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.fc_1_drop = nn.Dropout(self.hparams.drop_prob)

        self.fc_2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

        self.example_input_array = torch.zeros(20, 1, 20)
        
        
    def forward(self, x):
        """
        Just a simple net at the moment
        """
        x = self.fc_1(x.view(-1, self.hparams.in_features))
        x = torch.tanh(x)
        x = self.fc_1_bn(x)
        x = self.fc_1_drop(x)
        x = self.fc_2(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
        
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
        
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
        
        test_loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        return test_loss


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return [optimizer], [scheduler]
        return optimizer


    def setup(self, stage):
        all_data = RandomClasses(self.hparams.data_root)
        
        print("Length of all_data: ", len(all_data))
        print(self.hparams.num_data)
        
        self.validation_number = int(self.hparams.num_data * self.hparams.validation_portion)
        self.test_number = int(self.hparams.num_data * self.hparams.test_portion)
        self.training_number = self.hparams.num_data - self.test_number - self.validation_number
        
        print(self.training_number, self.validation_number, self.test_number, self.hparams.num_data)

        train_part, val_part, test_part = random_split(all_data, [self.training_number, self.validation_number, self.test_number])

        self.train_dataset = train_part
        self.val_dataset = val_part
        self.test_dataset = test_part

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, num_workers=self.hparams.num_workers)



class PoleClassifier(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the class of a complex structure in the vector
    """

    def __init__(self,
                 in_features: int = 64,
                 hidden_dim: int = 1000,
                 out_features: int = 2,
                 drop_prob: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 20,
                 **kwargs
                 ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        self.fc_1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.fc_1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.fc_1_drop = nn.Dropout(self.hparams.drop_prob)

        self.fc_2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.hidden_dim)
        self.fc_2_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.fc_2_drop = nn.Dropout(self.hparams.drop_prob)

        self.fc_3 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.hidden_dim)
        self.fc_3_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.fc_3_drop = nn.Dropout(self.hparams.drop_prob)

        self.fc_4 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

        self.example_input_array = torch.zeros(self.hparams.batch_size, self.hparams.in_features)
        
        
    def forward(self, x):
        """
        Just a simple net at the moment
        """
        # print("Testing the shape of the input: ", x.size())
        x = self.fc_1(x.view(-1, self.hparams.in_features))
        x = torch.relu(x)
        x = self.fc_1_bn(x)
        x = self.fc_1_drop(x)
        x = self.fc_2(x)
        x = torch.relu(x)
        x = self.fc_2_bn(x)
        x = self.fc_2_drop(x)
        x = self.fc_3(x)
        x = torch.relu(x)
        x = self.fc_3_bn(x)
        x = self.fc_3_drop(x)
        x = self.fc_4(x)
        # x = torch.argmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x).unsqueeze(-1)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
        
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x).unsqueeze(-1)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
                
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x).unsqueeze(-1)
        # print("y: ", y)
        # print("y_hat: ", y_hat)
        
        test_loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        return test_loss


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return [optimizer], [scheduler]
        return optimizer




class PoleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, validation_portion: float, test_portion: float):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_portion = validation_portion
        self.test_portion = test_portion
        
        seed_everything(1234)

    def setup(self, stage):
        all_data = PoleClasses(self.data_dir)
        
        num_data = len(all_data)
        print("Length of all_data: ", num_data)
        
        self.validation_number = int(num_data * self.validation_portion)
        self.test_number = int(num_data * self.test_portion)
        self.training_number = num_data - self.test_number - self.validation_number
        
        print("Data splits: ", self.training_number, self.validation_number, self.test_number, num_data)

        train_part, val_part, test_part = random_split(all_data, [self.training_number, self.validation_number, self.test_number])

        self.train_dataset = train_part
        self.val_dataset = val_part
        self.test_dataset = test_part

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

     
    
def objective(trial: optuna.trial.Trial):

    # Define those hyperparameters to be optimized
    # n_layers = trial.suggest_int("n_layers", 10, 14)
    dropout = trial.suggest_float("dropout", 0.0001, 0.1, log=True)
    # n_layers = 13 # trial.suggest_int("n_layers", 10, 14) 

    hidden_size = trial.suggest_int("hidden_size", 4, 128) # 4 up to 256
    batch_size = trial.suggest_int("batch_size", 4, 100, log=True)   # 4 up to 400
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-6, 1e-4, log=True)
    # learning_rate_init = 1e-4
    
    model = PoleClassifier(in_features = net_input_dim,
                 hidden_dim = hidden_size,
                 out_features = CLASSES,
                 drop_prob = dropout,
                 learning_rate = learning_rate_init,
                 batch_size = batch_size,
                 )
    
    datamodule = PoleDataModule(data_dir=basedir, batch_size=batch_size, 
                                validation_portion=VAL_PORTION, test_portion=TEST_PORTION)

    trainer = pl.Trainer(
        logger=tb_logger,
        # limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )


    # on production machine
    # trainer = pl.Trainer(gpus=1, distributed_backend='dp', max_epochs=EPOCHS, 
                        # logger=tb_logger, checkpoint_callback=False, log_gpu_memory='all',
                        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")])

    
    hyperparameters = dict(n_layers=3, learning_rate_init=learning_rate_init, 
                           batch_size=batch_size, hidden_size=hidden_size, 
                           dropout=dropout, net_input_dim=net_input_dim)
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


    return trainer.callback_metrics["val_loss"].item()




if __name__ == '__main__':

            
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if do_pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=NTRIALS, timeout=TIMEOUT)
    # study.optimize(objective, n_trials=NTRIALS, timeout=TIMEOUT, n_jobs=-1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    contour_params = ['batch_size', 'dropout']
    fig = optuna.visualization.plot_contour(study, params=[contour_params[0], contour_params[1]])
    filename_contour = "contour_plot_"+contour_params[0]+"_"+contour_params[1]+"_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig.write_image(filename_contour)

    contour_params = ['batch_size', 'hidden_size']
    fig = optuna.visualization.plot_contour(study, params=[contour_params[0], contour_params[1]])
    filename_contour = "contour_plot_"+contour_params[0]+"_"+contour_params[1]+"_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig.write_image(filename_contour)

    contour_params = ['hidden_size', 'learning_rate_init']
    fig = optuna.visualization.plot_contour(study, params=[contour_params[0], contour_params[1]])
    filename_contour = "contour_plot_"+contour_params[0]+"_"+contour_params[1]+"_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig.write_image(filename_contour)

    fig1 = optuna.visualization.plot_param_importances(study)

    filename_importance = "importance_plot_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig1.write_image(filename_importance)




