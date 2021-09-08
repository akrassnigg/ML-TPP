#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template: Training file

"""
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from parameters import train_portion_classifier, val_portion_classifier, test_portion_classifier
from parameters import architecture_classifier, hidden_dim_1_classifier, in_features_classifier, out_features_classifier
from parameters import weight_decay_classifier, batch_size_classifier, learning_rate_classifier, epochs_classifier
from parameters import data_dir_classifier, log_dir_classifier, models_dir_classifier
from lib.pole_classifier import Pole_Classifier, PoleDataModule_Classifier


##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':

    model_timestamp = int(time.time())

    tb_logger = TensorBoardLogger(log_dir_classifier, name='poles-classifier', version=model_timestamp)
    
    model = Pole_Classifier(
                weight_decay  = weight_decay_classifier,
                learning_rate = learning_rate_classifier,
                
                architecture = architecture_classifier,
                in_features  = in_features_classifier,
                out_features = out_features_classifier,
                
                hidden_dim_1  = hidden_dim_1_classifier
                )
                
    datamodule = PoleDataModule_Classifier(data_dir=data_dir_classifier, batch_size=batch_size_classifier, 
                                train_portion=train_portion_classifier, validation_portion=val_portion_classifier, test_portion=test_portion_classifier)
    
    checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
        dirpath = models_dir_classifier,
        filename = 'version_' + str(model_timestamp),
        monitor="val_acc",
        save_top_k=1, 
        mode='max',
        save_last= False
    )
    
    checkpoint_callback2 = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1, 
        mode='max',
        save_last= True
    )
    
    trainer = pl.Trainer(
        logger=tb_logger,
        val_check_interval=1,
        callbacks=[checkpoint_callback1, checkpoint_callback2],
        max_epochs=epochs_classifier,
        gpus=1
    )

    hyperparameters = dict(
                architecture = architecture_classifier,
        
                in_features  = in_features_classifier,
                out_features = out_features_classifier,
        
                hidden_dim_1 = hidden_dim_1_classifier,
                
                weight_decay = weight_decay_classifier,
                
                learning_rate = learning_rate_classifier,
                batch_size = batch_size_classifier)
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    


###############################################################################
###############################################################################
###############################################################################


#
