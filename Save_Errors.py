import numpy as np
from DLM import set_up_drift_discount_filter, set_up_local_discount_filter
from Paper_1 import load_data, filter_sample
from Utilities import load_data_all

def filter_and_save_err(Model, measure, save_label, verbose=True, all_runs=True):
     if all_runs:
          data = load_data_all(measure)
          print(f'Size of {measure} array: {len(data)}')
     else:
          data = load_data(measure)
     results = filter_sample(Model, data, 1, len(data), verbose=verbose)
     err = results.standardized_error()
     err[np.isnan(err)] = 0
     with open(f'{save_label}.npy', 'wb') as f:
          np.save(f, err)

if __name__ == '__main__':

     Models = {'v_x': set_up_local_discount_filter(0, omega=0.2618, df=0.70, alpha=2, beta=0.0001**2, J=3),
               'W2B0': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.85, alpha=2, beta=0.0001**2, J=4, my_EKF=True),
               'f_{plr}': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.94, alpha=2, beta=0.0001**2, J=4)}

     file_labels = {'v_x': 'xvelocity', 'W2B0': 'w2_b0', 'f_{plr}': 'percolate_left_right'}
     save_labels = {'v_x': 'vel_err', 'W2B0': 'w2_b0_err', 'f_{plr}': 'perc_err'}

     for key in ['v_x', 'W2B0', 'f_{plr}']:
          filter_and_save_err(Models[key], file_labels[key], save_labels[key])
     
     print('Done!')
