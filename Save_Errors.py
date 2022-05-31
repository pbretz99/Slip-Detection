import numpy as np
from DLM import set_up_drift_discount_filter, set_up_local_discount_filter
from Paper_1 import get_models, load_data, filter_sample

ModelVel = set_up_local_discount_filter(0, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3)
ModelW2B0 = set_up_drift_discount_filter(0.1, omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=4, my_EKF=True)
Vel = load_data('xvelocity')
W2 = load_data('w2_b0')

results_vel = filter_sample(ModelVel, Vel, 1, len(Vel))
results_W2 = filter_sample(ModelW2B0, W2, 1, len(W2))

vel_err = results_vel.standardized_error()
vel_err[np.isnan(vel_err)] = 0
with open('vel_err.npy', 'wb') as f:
     np.save(f, vel_err)

with open('vel_forecast.npy', 'wb') as f:
     np.save(f, results_vel.point_estimate(kind='forecast'))

w2_err = results_W2.standardized_error()
w2_err[np.isnan(w2_err)] = 0
with open('w2_b0_err.npy', 'wb') as f:
     np.save(f, w2_err)

with open('w2_b0_forecast.npy', 'wb') as f:
     np.save(f, results_W2.point_estimate(kind='forecast'))

print('Done!')
