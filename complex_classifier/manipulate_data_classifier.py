#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template: Training file

"""
from lib.training_data_generation_classifier import drop_classifier_samples_afterwards

from parameters import standard_re, fact_classifier, dst_min_classifier
from parameters import re_max, re_min, im_max, im_min
from parameters import coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import data_dir_classifier


##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':

    drop_classifier_samples_afterwards(with_mean=True, data_dir=data_dir_classifier, grid_x=standard_re, 
                                       re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                       coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                       coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                       fact=fact_classifier, dst_min=dst_min_classifier)
    


###############################################################################
###############################################################################
###############################################################################


#
