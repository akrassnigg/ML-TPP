#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template: Training file

"""
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import wandb
import argparse

from parameters import train_portion_classifier, val_portion_classifier, test_portion_classifier
from parameters import architecture_classifier
from parameters import hidden_dim_1_classifier, hidden_dim_2_classifier, hidden_dim_3_classifier
from parameters import hidden_dim_4_classifier, hidden_dim_5_classifier, hidden_dim_6_classifier
from parameters import in_features_classifier, out_features_classifier
from parameters import weight_decay_classifier, batch_size_classifier, learning_rate_classifier, epochs_classifier
from parameters import data_dir_classifier, log_dir_classifier, models_dir_classifier
from parameters import num_use_data_classifier, n_examples_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import fact_classifier, dst_min_classifier
from parameters import xtol
from parameters import num_runs_classifier
from parameters import optimizer_classifier
from parameters import drop_prob_1_classifier, drop_prob_2_classifier, drop_prob_3_classifier
from parameters import drop_prob_4_classifier, drop_prob_5_classifier, drop_prob_6_classifier
from lib.pole_classifier import Pole_Classifier, PoleDataModule_Classifier
from lib.plotting_routines import classifier_plot

##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':
    ##########   Parse hyperparameters, if wanted   ##########################
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hidden_dim_classifier', type=int, default=hidden_dim_1_classifier)
    parser.add_argument('--architecture_classifier', type=str, default=architecture_classifier)
    parser.add_argument('--batch_size_classifier', type=int, default=batch_size_classifier)
    parser.add_argument('--learning_rate_classifier', type=float, default=learning_rate_classifier)
    parser.add_argument('--optimizer_classifier', type=str, default=optimizer_classifier)
    parser.add_argument('--weight_decay_classifier', type=float, default=weight_decay_classifier)
    parser.add_argument('--drop_prob_classifier', type=float, default=drop_prob_1_classifier)
     
    args = parser.parse_args()
    hidden_dim_1_classifier  = args.hidden_dim_classifier
    hidden_dim_2_classifier  = args.hidden_dim_classifier
    hidden_dim_3_classifier  = args.hidden_dim_classifier
    hidden_dim_4_classifier  = args.hidden_dim_classifier
    hidden_dim_5_classifier  = args.hidden_dim_classifier
    hidden_dim_6_classifier  = args.hidden_dim_classifier
    architecture_classifier  = args.architecture_classifier
    batch_size_classifier    = args.batch_size_classifier
    learning_rate_classifier = args.learning_rate_classifier
    optimizer_classifier     = args.optimizer_classifier
    weight_decay_classifier  = args.weight_decay_classifier
    drop_prob_1_classifier   = args.drop_prob_classifier
    drop_prob_2_classifier   = args.drop_prob_classifier
    drop_prob_3_classifier   = args.drop_prob_classifier
    drop_prob_4_classifier   = args.drop_prob_classifier
    drop_prob_5_classifier   = args.drop_prob_classifier
    drop_prob_6_classifier   = args.drop_prob_classifier
    '''
    ##########################################################################
    
    net_hyperparameters = dict(
                architecture = architecture_classifier,
                in_features  = in_features_classifier,
                out_features = out_features_classifier, 
                hidden_dim_1 = hidden_dim_1_classifier,
                hidden_dim_2 = hidden_dim_2_classifier,
                hidden_dim_3 = hidden_dim_3_classifier,
                hidden_dim_4 = hidden_dim_4_classifier,
                hidden_dim_5 = hidden_dim_5_classifier,
                hidden_dim_6 = hidden_dim_6_classifier,
                
                weight_decay  = weight_decay_classifier,  
                drop_prob_1   = drop_prob_1_classifier,
                drop_prob_2   = drop_prob_2_classifier,
                drop_prob_3   = drop_prob_3_classifier,
                drop_prob_4   = drop_prob_4_classifier,
                drop_prob_5   = drop_prob_5_classifier,
                drop_prob_6   = drop_prob_6_classifier,
                
                learning_rate = learning_rate_classifier,
                optimizer     = optimizer_classifier
		)
    
    other_hyperparameters = dict(
                re_max = re_max, 
                re_min = re_min, 
                im_max = im_max, 
                im_min = im_min, 
                coeff_re_max = coeff_re_max, 
                coeff_re_min = coeff_re_min, 
                coeff_im_max = coeff_im_max, 
                coeff_im_min = coeff_im_min,

                fact_classifier = fact_classifier,
                dst_min_classifier = dst_min_classifier,
                xtol = xtol,
                
                n_examples_classifier = n_examples_classifier,
                num_use_data_classifier = num_use_data_classifier,
                train_portion_classifier = train_portion_classifier,
                val_portion_classifier = val_portion_classifier,
                test_portion_classifier = test_portion_classifier, 
                batch_size = batch_size_classifier
		)
    
    hyperparameters = {**net_hyperparameters, **other_hyperparameters}
    
    wandb.init(config=hyperparameters,
               entity="ml-tpp", project="pole_classifier",
               group="Experiment: retry fits ",
               notes="Classifier DataSet Experiment: In SciPy curve_fit: Set lower value for maxfev and retry fits 9 times, if they failed; with randomized p0",
               tags = ["Classifier", "DataSet Experiment"])

    logger = WandbLogger()  

    num_runs = num_runs_classifier
    test_acc_averaged = 0
    for i in range(num_runs): # average test_acc over multiple runs
        model = Pole_Classifier(
                    **net_hyperparameters
                    )
                    
        datamodule = PoleDataModule_Classifier(data_dir=data_dir_classifier, batch_size=batch_size_classifier, 
                                    train_portion=train_portion_classifier, validation_portion=val_portion_classifier, test_portion=test_portion_classifier, num_use_data=num_use_data_classifier)
        
        checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
            dirpath = models_dir_classifier,
            filename = 'name_' + str(wandb.run.name) + '_id_' + str(wandb.run.id) + '_' + str(i),
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
        
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=20, mode="max")
        
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=0.1,
            callbacks=[checkpoint_callback1, early_stop_callback], #, checkpoint_callback2
            max_epochs=epochs_classifier,
            gpus=1
        )
    
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        test_acc_averaged += trainer.logged_metrics["test_acc"].item()
    test_acc_averaged /= num_runs
    if num_runs > 1:
        wandb.log({'test_acc_averaged': test_acc_averaged})
    
    # Create classifier plot with test data
    test_dataloader = datamodule.test_dataloader()
    labels = test_dataloader.dataset[:][1]
    data_x = torch.from_numpy(test_dataloader.dataset[:][0])
    #preds  = trainer.predict(model, datamodule=datamodule)#, ckpt_path="best")
    model.to(device='cpu')
    preds = (torch.argmax(model(data_x), dim=1)).cpu().detach().numpy()
    fig, _ = classifier_plot(labels=labels, predictions=preds, do_return=True)
    wandb.log({"Classifier Plot": fig})

###############################################################################
###############################################################################
###############################################################################


#
