

import numpy as np
from Paper_1 import get_models, load_data, filter_sample

models = get_models()
Vel = load_data('xvelocity')
W2 = load_data('w2_b0')

results_vel = filter_sample(models[0], Vel, 1, len(Vel))
results_W2 = filter_sample(models[1], W2, 1, len(W2))

with open('vel_err.npy', 'wb') as f:
     np.save(f, results_vel.standardized_error())

with open('vel_forecast.npy', 'wb') as f:
     np.save(f, results_vel.point_estimate(kind='forecast'))

with open('w2_b0_err.npy', 'wb') as f:
     np.save(f, results_W2.standardized_error())

with open('w2_b0_forecast.npy', 'wb') as f:
     np.save(f, results_W2.point_estimate(kind='forecast'))
