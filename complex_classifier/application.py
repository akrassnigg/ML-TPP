#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Example: Apply SciPy curv_fit, classifier, NN regressor to data 

"""

from lib.scipy_fit_functions import get_all_scipy_preds
from lib.application_classifier_functions import get_classifier_preds
from lib.application_regressor_functions import get_all_regressor_preds
from lib.application_nnsc_functions import get_all_nnsc_preds

from parameters import dir_regressors, dir_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import regressor_subdirs
from parameters import xtol_classifier, p0_classifier, method_classifier, maxfev_classifier
from parameters import num_tries_classifier


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    # Step 0: Get Data 
    from test_data import my_p2, my_sigma_S, my_sigma_V
    grid_x = my_p2
    data_y = my_sigma_V

    # Do Scipy fit
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('SciPy fits:')
    params = get_all_scipy_preds(grid_x=grid_x, data_y=data_y, 
                                 re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                 coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                 coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                 with_bounds=True,
                                 p0='default', method='trf', maxfev=100000, num_tries=1, xtol = 1e-8)
    [print(paramsi) for paramsi in params]

    # Give out_re_pred to the classifier to see, if it correctly identifies it
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Classifier prediction:')
    class_pred      = get_classifier_preds(grid_x=grid_x, data_y=data_y,
                                           re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                           coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                           coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                           do_std=True, model_path=dir_classifier,
                                           with_bounds=True, p0=p0_classifier, 
                                           method=method_classifier, maxfev=maxfev_classifier, 
                                           num_tries=num_tries_classifier, xtol = xtol_classifier)
    print(class_pred)

    # Get Regressor predictions
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Regressor fits:')
    params_nn = get_all_regressor_preds(data_y=data_y, model_path=dir_regressors, regressor_subdirs=regressor_subdirs)
    [print(params_nni) for params_nni in params_nn]

    # Get NNSC preds
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Regressor + SciPy fits:')
    params_nnsc = get_all_nnsc_preds(grid_x=grid_x, data_y=data_y, 
                                     re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                     coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                     coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                     model_path=dir_regressors, regressor_subdirs=regressor_subdirs,
                                     with_bounds=True, method='trf', maxfev=100000, 
                                     num_tries=1, xtol = 1e-8)
    [print(params_nnsci) for params_nnsci in params_nnsc]

###############################################################################
###############################################################################
###############################################################################

                        

#
