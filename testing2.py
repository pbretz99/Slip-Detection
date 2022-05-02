
import pandas as pd
import numpy as np

from Utilities import load_data
from Times import get_times_from_error, print_measures_from_times, get_start_times

times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int)

Vel = load_data('xvelocity')
vel_err = np.load('vel_err.npy')

vel_detection, __ = get_times_from_error(vel_err, 1, 1, window_size=100)
vel_start = get_start_times(vel_detection, Vel, 0.001, window_size=5)

#w2_err = np.load('w2_b0_err.npy')
#times_w2, __ = get_times_from_error(w2_err, 1, 1, window_size=25)


print_measures_from_times(vel_start, times_df['Start'].to_numpy())
