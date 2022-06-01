import numpy as np
from DLM import set_up_drift_discount_filter, set_up_local_discount_filter
from Paper_1 import load_data, filter_sample

def filter_and_save_err(Model, measure, save_label, verbose=True):
     data = load_data(measure)
     results = filter_sample(Model, data, 1, len(data), verbose=verbose)
     err = results.standardized_error()
     err[np.isnan(err)] = 0
     with open(f'{save_label}.npy', 'wb') as f:
          np.save(f, err)

if __name__ == '__main__':

     Models = {'v_x': set_up_local_discount_filter(0, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3),
               'W2B0': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=4, my_EKF=True),
               'W2B1': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.75, alpha=2, beta=0.0001**2, J=4, my_EKF=True),
               'Perc': set_up_drift_discount_filter(0.1, omega=0.5466, df=0.85, alpha=2, beta=0.0001**2, J=3),
               'TP0': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3)}

     Models = {'v_x': set_up_local_discount_filter(0, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3),
               'W2B0': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=4, my_EKF=True),
               'W2B1': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.75, alpha=2, beta=0.0001**2, J=4, my_EKF=True),
               'Perc': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.88, alpha=2, beta=0.0001**2, J=4),
               'TP0': set_up_drift_discount_filter(0.1, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3)}

     #filter_and_save_err(Models['v_x'], 'xvelocity', 'vel_err')
     #filter_and_save_err(Models['W2B0'], 'w2_b0', 'w2_b0_err')
     #filter_and_save_err(Models['W2B1'], 'w2_b1', 'w2_b1_err')
     filter_and_save_err(Models['Perc'], 'percolate_left_right', 'perc_err')
     
     print('Done!')
