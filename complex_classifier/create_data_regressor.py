#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code: Generate data for the regressor

"""
from parameters import class_regressor, data_dir_regressor
from parameters import standard_re
from parameters import n_examples_regressor
from lib.training_data_generation_regressor import create_training_data_regressor


if __name__ == '__main__':

    create_training_data_regressor(length=n_examples_regressor, pole_class=class_regressor, mode='preparation', 
                         grid_x=standard_re, data_dir=data_dir_regressor)









#
