import numpy as np

from Paper_1 import get_models, get_vel_times, filter_sample, get_times_from_error
from Utilities import load_data_other_runs

models = get_models()
ModelVel, ModelW2 = models[0], models[1]

W2 = load_data_other_runs('w2_b0', dynamic=True)
Vel = load_data_other_runs('xvelocity', dynamic=True)


print('\nRunning W2\n')
results = filter_sample(ModelW2, W2, 30000, 35000)
#print(W2[34740:34750])
