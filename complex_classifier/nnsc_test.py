import numpy as np
from joblib import Parallel, delayed

from lib.get_params_functions import get_train_params_dual
from lib.curve_calc_functions import pole_curve_calc_dual
from lib.pole_config_organize import pole_config_organize_abs_dual
from lib.application_nnsc_functions import get_nnsc_pred
from lib.diverse_functions import drop_not_finite_rows

from parameters import class_regressor as pole_class
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import standard_re as grid_x
from parameters import dir_regressor

###############################################################################
# Step 1: Get Testing Data

length = 100000

# Generate pole configurations
params = get_train_params_dual(pole_class=pole_class, m=length, 
                          re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                          coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                          coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
# Calculate the pole curves
out_re = pole_curve_calc_dual(pole_class=pole_class, pole_params=params, grid_x=grid_x)
# Organize pole configurations
params = pole_config_organize_abs_dual(pole_class=pole_class, pole_params=params)

###############################################################################
# Step 2: Get NNSC predictions

def get_nnsc_pred_tmp(data_y_fun):
    return get_nnsc_pred(pole_class=pole_class, grid_x=grid_x, data_y=data_y_fun, 
                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                  model_path=dir_regressor, 
                  with_bounds=False, method='lm', maxfev=100000, 
                  num_tries=1, xtol = 1e-8)

tmp = Parallel(n_jobs=-1, backend="loky", verbose=10)(
             map(delayed(get_nnsc_pred_tmp), list(out_re)))


preds     = np.array([pred[0] for pred in tmp])
pred_type = np.array([pred[1] for pred in tmp])

unique, counts = np.unique(pred_type, return_counts=True)
pred_type_summary = dict(zip(unique, counts))
print('Summary of the used prediction methods:')
print(pred_type_summary)
f = open( 'Summary of the used prediction methods.txt', 'a' )
f.write( repr(pred_type_summary) + '\n' )
f.close()

preds = np.vstack(preds)

# Drop failed fits
tmp = len(out_re)
params, preds, out_re = drop_not_finite_rows(params, preds, out_re)
num_dropped = np.abs(len(out_re) - tmp)

###############################################################################
# Step 3: Calculate Error

# calculate Params RMSE of the fitting method:
params_rmse         = np.sqrt(np.mean((preds - params)**2, axis=0))
params_overall_rmse = np.sqrt(np.mean((preds - params)**2))

# calculate standard deviation of Params RMSE of the fitting method using bootstrapping:
num_bs = 1000
bs_indices                 = np.random.choice(np.arange(len(preds)), (num_bs, len(preds)), replace=True)
preds_bs                   = preds[bs_indices,:]
params_bs                  = params[bs_indices,:]
params_rmse_bs             = np.sqrt(np.mean((preds_bs - params_bs)**2, axis=1))
params_overall_rmse_bs     = np.sqrt(np.mean((preds_bs - params_bs)**2, axis=(1,2)))
params_rmse_bs_std         = np.std(params_rmse_bs, axis=0)
params_overall_rmse_bs_std = np.std(params_overall_rmse_bs, axis=0)

# write info about fits to txt file
dictionary = repr({
              'num_dropped': num_dropped,
              'params_rmse': params_rmse,
              'params_rmse_std': params_rmse_bs_std,
              'params_overall_rmse': params_overall_rmse,
              'params_overall_rmse_std': params_overall_rmse_bs_std
              })
f = open( 'nnsc+sc+nn_info.txt', 'a' )
f.write( dictionary + '\n' )
f.close()





