#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template
"""
import os
import sys
import pickle

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

from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import optuna
from optuna.trial import TrialState
from optuna.integration import PyTorchLightningPruningCallback

do_pruning = False

EPOCHS = None
TIMEOUT = None
NTRIALS = None
TRAIN_SAMPLES = 200000
VAL_SAMPLES = 20000
TEST_SAMPLES = 20000
CLASSES = 9

basedir = './data'
logdir = "logs"


class PoleClasses(Dataset):
    def __init__(self, data_dir):

        self.data_X = np.load(data_dir+"/various_poles_data_classifier_x.npy", allow_pickle=True).astype('float32')
        self.data_y = np.reshape(np.load(data_dir+"/various_poles_data_classifier_y.npy", allow_pickle=True).astype('int64'), (-1,1))
        
        print("Checking shape of loaded data: X: ", np.shape(self.data_X))
        print("Checking shape of loaded data: y: ", np.shape(self.data_y))

    def __len__(self):
        num_of_data_points = len(self.data_X)
        print("Successfully loaded pole training data of length: ", num_of_data_points)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]
    
    
net_input_dim = len(PoleClasses(basedir)[0][0])


class PoleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, train_portion: float, validation_portion: float, test_portion: float):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.test_portion = test_portion
        
        seed_everything(1234)

    def setup(self, stage):
        all_data = PoleClasses(self.data_dir)
        
        num_data = len(all_data)
        print("Length of all_data: ", num_data)
        
        self.training_number = self.train_portion
        self.validation_number = self.validation_portion
        self.test_number = self.test_portion
        
        print("Data splits: ", self.training_number, self.validation_number, self.test_number, num_data)

        train_part, val_part, test_part = random_split(all_data, [self.training_number, self.validation_number, self.test_number])

        self.train_dataset = train_part
        self.val_dataset = val_part
        self.test_dataset = test_part

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last = True) #need drop_last=True for BatchNorm1D to work for any Batchsize

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.validation_number, drop_last = True) #need drop_last=True for BatchNorm1D to work for any Batchsize

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_number, drop_last = True) #need drop_last=True for BatchNorm1D to work for any Batchsize


class PoleClassifier(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the class of a complex structure in the vector
    """

    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int  = net_input_dim,
                 out_features: int = CLASSES,
 
                 ### Fully connected layers
                 hidden_dim1: int = 1000, 
                 hidden_dim2: int = 1000, 
                 hidden_dim3: int = 1000, 
                 hidden_dim4: int = 1000, 
                 
                 ### Regularization
                 weight_decay: float = 0.0,
                 
                 ### Learning parameters
                 learning_rate: float = 0.001,
                 batch_size:      int = 20,
                 
                 **kwargs
                 ):
        # init superclasss
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()
        
        self.initial_bn = nn.BatchNorm1d(self.hparams.in_features)

        self.fc1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim1)
        self.fc1_bn = nn.BatchNorm1d(self.hparams.hidden_dim1)
        
        self.fc2 = nn.Linear(in_features=self.hparams.hidden_dim1,
                              out_features=self.hparams.hidden_dim2)
        self.fc2_bn = nn.BatchNorm1d(self.hparams.hidden_dim2)
        
        self.fc3 = nn.Linear(in_features=self.hparams.hidden_dim2,
                              out_features=self.hparams.hidden_dim3)
        self.fc3_bn = nn.BatchNorm1d(self.hparams.hidden_dim3)
        
        self.fc4 = nn.Linear(in_features=self.hparams.hidden_dim3,
                              out_features=self.hparams.hidden_dim4)
        self.fc4_bn = nn.BatchNorm1d(self.hparams.hidden_dim4)
        
        self.fc_out = nn.Linear(in_features=self.hparams.hidden_dim4,
                              out_features=self.hparams.out_features)
        
        
    def forward(self, x):        
        x = self.initial_bn(x)
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc1_bn(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc2_bn(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc3_bn(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc4_bn(x)
        
        x = self.fc_out(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1)
        
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
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


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}


 
    
def objective(trial: optuna.trial.Trial):
    model_timestamp = int(time.time())

    tb_logger = TensorBoardLogger(logdir, name='ml-tpp-classifier', version=model_timestamp)

    ### Dataset-specific parameters
    in_features  = net_input_dim
    out_features = CLASSES

    ### Fully connected layers
    hidden_dim1 = 256#trial.suggest_int("hidden_dim1", 4, 1024, log=False)
    hidden_dim2 = 128#trial.suggest_int("hidden_dim2", 4, 1024, log=False)
    hidden_dim3 = 64#trial.suggest_int("hidden_dim3", 4, 1024, log=False)
    hidden_dim4 = 32#trial.suggest_int("hidden_dim3", 4, 1024, log=False)
    
    ### Regularization
    weight_decay = 0.0#trial.suggest_float('weight_decay', 1e-1, 1e0, log=False)
                 
    #Training hparams
    batch_size         = 200000#trial.suggest_int("batch_size", 512, 10000, log=False)  
    learning_rate_init = 1e-3#4#4#5#trial.suggest_float('learning_rate_init', 1e-6, 1e-1, log=True) 

    model = PoleClassifier(
                in_features = in_features,
                out_features = out_features,
                
                hidden_dim1 = hidden_dim1,
                hidden_dim2 = hidden_dim2,
                hidden_dim3 = hidden_dim3,
                hidden_dim4 = hidden_dim4,
                
                weight_decay = weight_decay,
                
                learning_rate = learning_rate_init,
                batch_size = batch_size
                )
    
    datamodule = PoleDataModule(data_dir=basedir, batch_size=batch_size, 
                                train_portion=TRAIN_SAMPLES, validation_portion=VAL_SAMPLES, test_portion=TEST_SAMPLES)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1, 
        mode='max',
        save_last= True
    )
    
    #early_stopping = EarlyStopping('val_acc', mode='max', patience=1000)
    
    trainer = pl.Trainer(
        logger=tb_logger,
        #val_check_interval=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        #callbacks=[early_stopping, PyTorchLightningPruningCallback(trial, monitor="val_acc")], 
        gpus=1
    )

    hyperparameters = dict(
                hidden_dim1 = hidden_dim1,
                hidden_dim2 = hidden_dim2,
                hidden_dim3 = hidden_dim3,
                hidden_dim4 = hidden_dim4,
                
                weight_decay = weight_decay,
                
                learning_rate = learning_rate_init,
                batch_size = batch_size)
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    return trainer.callback_metrics["test_acc"].item()




if __name__ == '__main__':
    #'''
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if do_pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=NTRIALS, timeout=TIMEOUT)
    # study.optimize(objective, n_trials=NTRIALS, timeout=TIMEOUT, n_jobs=-1)
    pickle.dump(study, open( "study.p", "wb" ) )
    #study = pickle.load( open( "study.p", "rb" ) )
    
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


    contour_params = ['batch_size', 'weight_decay']
    fig = optuna.visualization.plot_contour(study, params=[contour_params[0], contour_params[1]])
    filename_contour = "contour_plot_"+contour_params[0]+"_"+contour_params[1]+"_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig.write_image(filename_contour)
    
    fig1 = optuna.visualization.plot_param_importances(study)

    filename_importance = "importance_plot_numtrials_"+str(NTRIALS)+"_epochs_"+str(EPOCHS)+".png"
    fig1.write_image(filename_importance)
    
    ###########################################################################
    ###########################################################################
    '''
    model_timestamp = int(time.time())

    tb_logger = TensorBoardLogger(logdir, name='ml-tpp-classifier', version=model_timestamp)
             
    #Training hparams
    batch_size    = 200000
    learning_rate = 1e-3
    
    datamodule = PoleDataModule(data_dir=basedir, batch_size=batch_size, 
                                train_portion=TRAIN_SAMPLES, validation_portion=VAL_SAMPLES, test_portion=TEST_SAMPLES)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1, 
        mode='max',
        save_last= True
    )
    
    #early_stopping = EarlyStopping('val_acc', mode='max', patience=1000)

    trainer = pl.Trainer(resume_from_checkpoint="./best.ckpt", 
                        logger=tb_logger,
                        #val_check_interval=1,
                        checkpoint_callback=checkpoint_callback,
                        max_epochs=EPOCHS,
                        #callbacks=[early_stopping],
                        gpus=1
                        )
    
    model = PoleClassifier.load_from_checkpoint("./best.ckpt",
                                                batch_size=batch_size,
                                                learning_rate=learning_rate)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    #'''

###############################################################################
###############################################################################
###############################################################################


#