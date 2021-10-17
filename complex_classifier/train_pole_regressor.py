#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template: Training file
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger  

from lib.pole_regressor import Pole_Regressor, PoleDataModule_Regressor

from parameters import class_regressor
from parameters import standard_re
from parameters import data_dir_regressor, log_dir_regressor, models_dir_regressor
from parameters import train_portion_regressor, val_portion_regressor, test_portion_regressor
from parameters import architecture_regressor, in_features_regressor, out_features_regressor
from parameters import hidden_dim_1_regressor, hidden_dim_2_regressor, hidden_dim_3_regressor, hidden_dim_4_regressor, hidden_dim_5_regressor, hidden_dim_6_regressor
from parameters import epochs_regressor, learning_rate_regressor, weight_decay_regressor
from parameters import batch_size_regressor, training_step_regressor
from parameters import num_use_data_regressor
from parameters import drop_prob_1_regressor, drop_prob_2_regressor, drop_prob_3_regressor
from parameters import drop_prob_4_regressor, drop_prob_5_regressor, drop_prob_6_regressor
from parameters import optimizer_regressor
from parameters import num_epochs_use_regressor
from parameters import val_check_interval_regressor, es_patience_regressor
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import fact_regressor, dst_min_regressor
from parameters import n_examples_regressor
from parameters import num_runs_regressor
from parameters import name_ckpt_regressor
from parameters import class_regressor
from parameters import loss_name_regressor


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    
    net_hyperparameters = dict(
                pole_class = class_regressor,
                
                std_path = data_dir_regressor,
        
                re_max = re_max, 
                re_min = re_min, 
                im_max = im_max, 
                im_min = im_min, 
                coeff_re_max = coeff_re_max, 
                coeff_re_min = coeff_re_min, 
                coeff_im_max = coeff_im_max, 
                coeff_im_min = coeff_im_min,
                
                grid_x=standard_re,
                
                architecture = architecture_regressor,
                in_features  = in_features_regressor,
                out_features = out_features_regressor, 
                hidden_dim_1 = hidden_dim_1_regressor,
                hidden_dim_2 = hidden_dim_2_regressor,
                hidden_dim_3 = hidden_dim_3_regressor,
                hidden_dim_4 = hidden_dim_4_regressor,
                hidden_dim_5 = hidden_dim_5_regressor,
                hidden_dim_6 = hidden_dim_6_regressor,
                
                weight_decay  = weight_decay_regressor,  
                drop_prob_1   = drop_prob_1_regressor,
                drop_prob_2   = drop_prob_2_regressor,
                drop_prob_3   = drop_prob_3_regressor,
                drop_prob_4   = drop_prob_4_regressor,
                drop_prob_5   = drop_prob_5_regressor,
                drop_prob_6   = drop_prob_6_regressor,
                
                learning_rate = learning_rate_regressor,
                optimizer     = optimizer_regressor
		)
    
    other_hyperparameters = dict(
                fact_regressor    = fact_regressor,
                dst_min_regressor = dst_min_regressor,
                
                n_examples_regressor    = n_examples_regressor,
                num_use_data_regressor  = num_use_data_regressor,
                train_portion_regressor = train_portion_regressor,
                val_portion_regressor   = val_portion_regressor,
                test_portion_regressor  = test_portion_regressor, 
                batch_size               = batch_size_regressor,
                
                val_check_interval = val_check_interval_regressor,
                es_patience        = es_patience_regressor,

                loss = loss_name_regressor
		)
    
    hyperparameters = {**net_hyperparameters, **other_hyperparameters}
    
    wandb.init(config=hyperparameters,
               entity="ml-tpp", project="pole_classifier",
               group="Regressor Experiment: Reconstruction Loss",
               notes=" ",
               tags = ["Regressor"])

    logger = WandbLogger() 
    
    num_runs = num_runs_regressor
    test_loss_averaged = 0
    for i in range(num_runs): # average test_acc over multiple runs
        if training_step_regressor == 0: #start training from scratch
            model = Pole_Regressor(
                        **net_hyperparameters
                        )
            
        elif training_step_regressor == 1: # resume training          
            model = Pole_Regressor.load_from_checkpoint(models_dir_regressor + name_ckpt_regressor,
                                                        **net_hyperparameters)
            
        datamodule = PoleDataModule_Regressor(pole_class=class_regressor, grid_x=standard_re, data_dir=data_dir_regressor, batch_size=batch_size_regressor, 
                                train_portion=train_portion_regressor, validation_portion=val_portion_regressor, test_portion=test_portion_regressor, 
                                num_use_data=num_use_data_regressor, num_epochs_use=num_epochs_use_regressor,
                                fact=fact_regressor, dst_min=dst_min_regressor,
                                re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        
        checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
            dirpath = models_dir_regressor,
            filename = 'name_' + str(wandb.run.name) + '_id_' + str(wandb.run.id) + '_' + str(i),
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
        
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=es_patience_regressor, mode="min")
        
        
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=val_check_interval_regressor,
            callbacks=[checkpoint_callback1, early_stop_callback], #, checkpoint_callback2
            max_epochs=epochs_regressor,
            gpus=1
        )
    
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        test_loss_averaged += trainer.logged_metrics["test_loss"].item()
    test_loss_averaged /= num_runs
    if num_runs > 1:
        wandb.log({'test_loss_averaged': test_loss_averaged})
        
        
        
        
        
    









#