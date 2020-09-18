#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@author: andreaskrassnigg

Classifier based on pytorch basic template
"""


import os
from argparse import ArgumentParser

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


logdir = "logs"

model_timestamp = int(time.time())

tb_logger = TensorBoardLogger(logdir, name='ml-tpp-classifier', version=model_timestamp)

class RandomClasses(Dataset):
    def __init__(self):

        self.data_X = np.load("random_test_data_classifier_x.npy", allow_pickle=True).astype('float32')
        self.data_y = np.load("random_test_data_classifier_y.npy", allow_pickle=True).astype('int')

    def __len__(self):
        num_of_data_points = len(self.data_X)
        print("Successfully loaded training data of length: ", num_of_data_points)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]
    



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
                 data_root: str = './datasets',
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
        """
        Training step takes batch for training
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        print("y: ", y)
        print("y_hat: ", y_hat)
        
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Same as training, just evaluates a little differently
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        """
        Same as training, just evaluates a little differently
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

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

    # def prepare_data(self):
    #     MNIST(self.hparams.data_root, train=True, download=True, transform=transforms.ToTensor())
    #     MNIST(self.hparams.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        all_data = RandomClasses()
        
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

    # @staticmethod
    # def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
    #     """
    #     Define parameters that only apply to this model
    #     """
    #     parser = ArgumentParser(parents=[parent_parser])

    #     # param overwrites
    #     # parser.set_defaults(gradient_clip_val=5.0)

    #     # network params
    #     parser.add_argument('--in_features', default=28 * 28, type=int)
    #     parser.add_argument('--hidden_dim', default=50000, type=int)
    #     # use 500 for CPU, 50000 for GPU to see speed difference
    #     parser.add_argument('--out_features', default=10, type=int)
    #     parser.add_argument('--drop_prob', default=0.2, type=float)

    #     # data
    #     parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)
    #     parser.add_argument('--num_workers', default=4, type=int)

    #     # training params (opt)
    #     parser.add_argument('--epochs', default=20, type=int)
    #     parser.add_argument('--batch_size', default=64, type=int)
    #     parser.add_argument('--learning_rate', default=0.001, type=float)
    #     return parser


if __name__ == '__main__':

            
    # locally, for testing, debugging, etc.: fast_dev_run=True, 
    trainer = pl.Trainer(max_epochs=3, logger=tb_logger)

    # on production machimne
    # trainer = pl.Trainer(gpus=1, distributed_backend='dp', max_epochs=30, logger=tb_logger, log_gpu_memory='all')


    # run the training, etc., including validation on part of the training data set:
    model = DeepNet1(in_features = 20,
                 hidden_dim = 100,
                 out_features = 4,
                 drop_prob = 0.2,
                 learning_rate = 0.001 * 8,
                 batch_size = 200,
                 val_batch_size = 40,
                 test_batch_size = 40,
                 validation_portion = .2,
                 test_portion = .2,
                 data_root = '.',
                 num_data = 200,
                 num_workers = 8)
                 
    trainer.fit(model)
    trainer.test(model)

