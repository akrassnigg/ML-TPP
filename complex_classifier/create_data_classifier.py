#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Generate data for the classifier

"""
import time

from parameters import n_examples_classifier, standard_re, data_dir_classifier
from lib.training_data_generation_classifier import create_training_data_classifier

if __name__ == '__main__':
    time1 = time.time()
    create_training_data_classifier(length=n_examples_classifier, grid_x=standard_re, with_bounds=True, data_dir=data_dir_classifier)
    print(time.time() - time1)





