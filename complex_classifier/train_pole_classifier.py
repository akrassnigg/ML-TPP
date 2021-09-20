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

from parameters import train_portion_classifier, val_portion_classifier, test_portion_classifier
from parameters import architecture_classifier, hidden_dim_1_classifier, in_features_classifier, out_features_classifier
from parameters import weight_decay_classifier, batch_size_classifier, learning_rate_classifier, epochs_classifier
from parameters import data_dir_classifier, log_dir_classifier, models_dir_classifier
from parameters import experimental_num_use_data_classifier, n_examples_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import fact_classifier, dst_min_classifier
from lib.pole_classifier import Pole_Classifier, PoleDataModule_Classifier
from lib.plotting_routines import classifier_plot
import wandb

##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':

    model_timestamp = int(time.time())
    
    logger = WandbLogger(entity="ml-tpp", project="pole_classifier",
                         group="Experiment: drop small poles relative",
                         notes="Classifier DataSet Experiment: Drop samples with poles that are factors smaller than other poles and compare test_acc for different values of fact_classifier",
                         tags = ["Classifier", "DataSet Experiment"])
    
    model = Pole_Classifier(
                weight_decay  = weight_decay_classifier,
                learning_rate = learning_rate_classifier,
                
                architecture = architecture_classifier,
                in_features  = in_features_classifier,
                out_features = out_features_classifier,
                
                hidden_dim_1  = hidden_dim_1_classifier
                )
                
    datamodule = PoleDataModule_Classifier(data_dir=data_dir_classifier, batch_size=batch_size_classifier, 
                                train_portion=train_portion_classifier, validation_portion=val_portion_classifier, test_portion=test_portion_classifier, num_use_data=experimental_num_use_data_classifier)
    
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
    
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=10, mode="max")
    
    trainer = pl.Trainer(
        logger=logger,
        #val_check_interval=1,
        callbacks=[checkpoint_callback1, checkpoint_callback2, early_stop_callback],
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
                batch_size = batch_size_classifier,
                
                re_max = re_max, 
                re_min = re_min, 
                im_max = im_max, 
                im_min = im_min, 
                coeff_re_max = coeff_re_max, 
                coeff_re_min = coeff_re_min, 
                coeff_im_max = coeff_im_max, 
                coeff_im_min = coeff_im_min,
                n_examples_classifier = n_examples_classifier,
                experimental_num_use_data_classifier = experimental_num_use_data_classifier,
                fact_classifier = fact_classifier,
                dst_min_classifier = dst_min_classifier,
                train_portion_classifier = train_portion_classifier,
                val_portion_classifier = val_portion_classifier,
                test_portion_classifier = test_portion_classifier       
		)

    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
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
