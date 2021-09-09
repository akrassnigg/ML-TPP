#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template: Training file
"""
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning.callbacks import EarlyStopping

from parameters import class_regressor
from parameters import standard_re
from parameters import data_dir_regressor, log_dir_regressor, models_dir_regressor
from parameters import train_portion_regressor, val_portion_regressor, test_portion_regressor
from parameters import architecture_regressor, in_features_regressor, out_features_regressor
from parameters import hidden_dim_1_regressor, hidden_dim_2_regressor, hidden_dim_3_regressor, hidden_dim_4_regressor, hidden_dim_5_regressor, hidden_dim_6_regressor
from parameters import epochs_regressor, learning_rate_regressor, weight_decay_regressor
from parameters import batch_size_regressor, val_check_interval_regressor, training_step_regressor
from lib.pole_regressor import Pole_Regressor, PoleDataModule_Regressor



##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    model_timestamp = int(time.time())    
    logger = WandbLogger(project="test_project")
    
    ### Dataset-specific parameters
    in_features  = in_features_regressor
    out_features = out_features_regressor
    
    if training_step_regressor == 0:        
        batch_size         = batch_size_regressor
        val_check_interval = val_check_interval_regressor
        
        model = Pole_Regressor(
                    weight_decay  = weight_decay_regressor,
                    learning_rate = learning_rate_regressor,
                    
                    architecture = architecture_regressor,
                    in_features  = in_features_regressor,
                    out_features = out_features_regressor,
                    
                    hidden_dim_1  = hidden_dim_1_regressor,
                    hidden_dim_2  = hidden_dim_2_regressor,
                    hidden_dim_3  = hidden_dim_3_regressor,
                    hidden_dim_4  = hidden_dim_4_regressor,
                    hidden_dim_5  = hidden_dim_5_regressor,
                    hidden_dim_6  = hidden_dim_6_regressor
                    )
        
    elif training_step_regressor == 1:            
        batch_size         = batch_size_regressor
        val_check_interval = val_check_interval_regressor
        
        model = Pole_Regressor.load_from_checkpoint("./regressor_ckpt.ckpt",
                                                    learning_rate=learning_rate_regressor,
                                                    weight_decay=weight_decay_regressor)
        
        
    datamodule = PoleDataModule_Regressor(pole_class=class_regressor, grid_x=standard_re, data_dir=data_dir_regressor, batch_size=batch_size_regressor, 
                            train_portion=train_portion_regressor, validation_portion=val_portion_regressor, test_portion=test_portion_regressor)
    
    checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
        dirpath = models_dir_regressor,
        filename = 'version_' + str(model_timestamp),
        monitor="val_loss",
        save_top_k=1, 
        mode='min',
        save_last= False
    )
    
    checkpoint_callback2 = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1, 
        mode='min',
        save_last= True
    )
    
    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback1, checkpoint_callback2],
        max_epochs=epochs_regressor,
        gpus=1
    )
    
    hyperparameters = dict(
            architecture  = architecture_regressor,
            in_features   = in_features_regressor,
            out_features  = out_features_regressor,      
            hidden_dim_1  = hidden_dim_1_regressor,
            hidden_dim_2  = hidden_dim_2_regressor,
            hidden_dim_3  = hidden_dim_3_regressor,
            hidden_dim_4  = hidden_dim_4_regressor,
            hidden_dim_5  = hidden_dim_5_regressor,
            hidden_dim_6  = hidden_dim_6_regressor,
            weight_decay  = weight_decay_regressor,
            learning_rate = learning_rate_regressor,
            batch_size    = batch_size_regressor)
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")









#