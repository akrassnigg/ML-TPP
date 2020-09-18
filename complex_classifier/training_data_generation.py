#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code

"""


import numpy as np
import matplotlib.pyplot as plt



n_examples = 200

def single_complex_pole(list_re, list_im, part_re, part_im, power=1, coeff=1.):
    """
    Computes the function values on a list of complex coordinates of a
    single singularity of power power, with a multiplicative coefficient coeff
    and a position in the complex plane given by part_re and part_im
    
    """
    diff = ((np.array(list_re) - np.ones(len(list_re)) * part_re) + 
        1j * (np.array(list_im) - np.ones(len(list_im)) * part_im) )
    
    result = coeff/diff**power
    
    result_re = np.real(result)
    
    result_im = np.imag(result)
    
    return result_re, result_im


def list_of_random_vectors(n_vectors, length_vectors):
    return np.reshape(np.random.random(size=n_vectors*length_vectors), (n_vectors, length_vectors))


# real_axis_sample = np.linspace(0,10,num=20)

# testvals = single_complex_pole(real_axis_sample, np.zeros(len(real_axis_sample)), -1., 0.)

# print(testvals)

# create and save testing training data

test_random_data_x = list_of_random_vectors(n_examples, 20)
test_random_data_y = np.random.randint(0,high=3, size=n_examples)


np.save("random_test_data_classifier_x.npy", test_random_data_x)
np.save("random_test_data_classifier_y.npy", test_random_data_y)
