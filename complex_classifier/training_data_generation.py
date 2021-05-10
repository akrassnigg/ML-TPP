#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code

"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd



n_examples = 240000

the_scale = 10.0

#standard_re = np.linspace(0.01, 20., num=64)
standard_re = pd.read_csv("./integration_gridpoints.csv").to_numpy().reshape(-1)
standard_im = np.linspace(0., 0., num=64)

def single_complex_pole(list_re, list_im, part_re, part_im, power=1, coeff=1.):
    """
    Computes the function values on a list of complex coordinates of a
    single singularity of power power, with a multiplicative complex coefficient coeff
    and a position in the complex plane given by part_re and part_im
    
    """
    diff = ((np.array(list_re) - np.ones(len(list_re)) * part_re) + 
        1j * (np.array(list_im) - np.ones(len(list_im)) * part_im) )
    
    result = coeff/diff**power
    
    result_re = np.real(result)
    
    result_im = np.imag(result)
    
    return result_re, result_im


def single_real_pole(list_re, part_re, power=1, coeff=1.):
    """
    Computes the function values on a list of coordinates on the real axis of a
    single singularity of power power, with a multiplicative coefficient coeff
    and a position on the real axis given by part_re 
    
    """
    
    len_set = len(list_re)
    list_im = np.linspace(0., 0., num=len_set)
    
    result_re, _ = single_complex_pole(list_re, list_im, part_re, 0., power=power, coeff=coeff)
    
    return result_re


def complex_conjugate_pole_pair(list_re, part_re, part_im, power=1, coeff=1.):
    """
    Computes the function values on a list of coordinates on the real axis of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative coefficient coeff and a position on the real axis 
    given by part_re and on the imaginary axis given by part_im 
    
    """
    
    len_set = len(list_re)
    list_im = np.linspace(0., 0., num=len_set)
    
    result_re_plus, _ = single_complex_pole(list_re, list_im, part_re, part_im, power=power, coeff=coeff)
    result_re_minus, _ = single_complex_pole(list_re, list_im, part_re, - part_im, power=power, coeff=np.conjugate(coeff))
    
    return result_re_plus + result_re_minus


def list_of_random_vectors(n_vectors, length_vectors):
    return np.reshape(np.random.random(size=n_vectors*length_vectors), (n_vectors, length_vectors))


# real_axis_sample = np.linspace(0,10,num=20)

# testvals = single_complex_pole(real_axis_sample, np.zeros(len(real_axis_sample)), -1., 0.)

# print(testvals)

# create and save testing training data


def create_training_data(length=n_examples):
    
    # # Random example numbers
    # test_random_data_x = list_of_random_vectors(n_examples, 20)
    # test_random_data_y = np.random.randint(0,high=3, size=n_examples)
    
    
    # np.save("random_test_data_classifier_x.npy", test_random_data_x)
    # np.save("random_test_data_classifier_y.npy", test_random_data_y)

    data_x = []
    data_y = []
    
    for counter in range(n_examples):
        label = np.random.choice([0,1,2,3,4,5,6,7,8])
        
        if label == 0:
            # a single pole on the real axis
            
            part_re = the_scale * np.random.random()
            
            out_re = single_real_pole(standard_re, part_re)
            
        elif label == 1:
            # a complex conjugated pole pair
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)
            
            out_re = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
        elif label == 2:
            # two real poles
            
            part_re = the_scale * np.random.random()
            out_re1 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * np.random.random()
            out_re2 = single_real_pole(standard_re, part_re)
            
            out_re  = out_re1 + out_re2
            
        elif label == 3:
            # a real pole and a cc pole pair
            
            #Real pole
            part_re = the_scale * np.random.random()
            out_re1 = single_real_pole(standard_re, part_re)
            
            #cc pole pair
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re2 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            out_re  = out_re1 + out_re2
            
        elif label == 4:
            # two cc pole pairs
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re1 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re2 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            out_re  = out_re1 + out_re2
            
        elif label == 5:
            # three real poles
            
            part_re = the_scale * np.random.random()
            out_re1 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * np.random.random()
            out_re2 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * np.random.random()
            out_re3 = single_real_pole(standard_re, part_re)
            
            out_re  = out_re1 + out_re2 + out_re3
            
        elif label == 6:
            # two real poles and one cc pole pair
            
            part_re = the_scale * np.random.random()
            out_re1 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * np.random.random()
            out_re2 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re3 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            out_re  = out_re1 + out_re2 + out_re3
            
        elif label == 7:
            # one real pole and two cc pole pairs
            
            part_re = the_scale * np.random.random()
            out_re1 = single_real_pole(standard_re, part_re)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re2 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re3 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            out_re  = out_re1 + out_re2 + out_re3
            
        elif label == 8:
            # three cc pole pairs
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re1 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re2 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            part_re = the_scale * 2 * (np.random.random() - 0.5)
            part_im = the_scale * 2 * (np.random.random() - 0.5)   
            out_re3 = complex_conjugate_pole_pair(standard_re, part_re, part_im)
            
            out_re  = out_re1 + out_re2 + out_re3
            
        else:
            # should not happen
            sys.exit("Undefined label.")
        
        data_x.append(out_re)
        data_y.append(label)

    np.save("various_poles_data_classifier_x.npy", data_x)
    np.save("various_poles_data_classifier_y.npy", data_y)

    print("Successfully saved x data of shape ", np.shape(data_x))
    print("Successfully saved y data of shape ", np.shape(data_y))

    return


if __name__ == '__main__':

    # out_re, out_im = single_complex_pole(standard_re, standard_im, 1.3, 0.)     

    # print("Re: ", out_re)
    # print("Im: ", out_im)


    create_training_data(length=n_examples)









#
