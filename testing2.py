

from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from Paper_1 import get_models
from Utilities import print_tracker, load_data

def train_model(Model, Data, init, final, set_init=True, verbose=False):
     if set_init: Model.m[0,0] = Data[init]
     results = np.zeros((final-init,))
     for t in range(init, final):
          if verbose: print_tracker(t-init, final-init)
          ret = Model.filter(Data[t], return_results=True)
          results[t-init] = ret['forecast']
     if verbose: print('Complete!')
     return results

'''
W2 = load_data('w2_b0')
model = get_models()[1]
init, break_point, final = 6000, 6100, 6250
forecast = train_model(model, W2, init, break_point)
print(model.m.flatten()[0:2])
sample = model.monte_carlo(steps=final-break_point)
print(len(sample))
print(final-break_point)

fig, ax = plt.subplots(figsize=(5, 10))
ax.plot(range(init, final), W2[init:final], c='gray')
ax.plot(range(init, break_point), forecast, c='steelblue')
ax.plot(range(break_point, final), sample, c='green')
plt.show()
'''


import pandas as pd
import numpy as np

from Utilities import load_data, print_tracker
from Times import get_times_from_error, get_all_measures_from_times, print_measures_from_times

def my_measures(vel_detection, slip_start):
     __, __, __, counts = get_all_measures_from_times(vel_detection, slip_start)
     N_slips, N_detections, N_matched, N_just_missed = counts['Labels'], counts['Detections'], counts['Matched'], counts['Just Missed']
     if N_detections == 0:
          return np.array([0, 0, 0])
     f_p = 1 - N_matched / N_detections
     t_p_total = N_matched / N_slips
     t_p_partial = (N_matched - N_just_missed) / N_slips
     return np.array([f_p, t_p_total, t_p_partial])

def accuracy_measures(slip_start, eps_range, err, verbose=False):
     measures = np.zeros((len(eps_range), 3))
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          detection, __ = get_times_from_error(err, 1, eps_range[i], window_size=25)
          measures[i] = my_measures(detection, slip_start)
     return measures

def plot_accuracy_measures(ax, measures, eps_range, data_label):
     measure_labels = ['$f_p$', '$t_p$ (total)', '$t_p$ (partial)']
     measure_colors = ['orange', 'steelblue', 'steelblue']
     measure_ls = ['-', '--', '-']
     for i in range(3):
          ax.plot(eps_range, measures[:,i], label=measure_labels[i], c=measure_colors[i], ls=measure_ls[i])
     ax.axhline(y=1, c='lightgray', ls='--')
     ax.set_ylim(bottom=0)
     ax.legend()
     ax.set_xlabel('Error Threshold $\epsilon$')
     ax.set_ylabel('Rate')
     ax.set_title(f'Accuracy Measures for {data_label} Detections')

times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int)

#Vel = load_data('xvelocity')
vel_err = np.load('vel_err.npy')
w2_err = np.load('w2_b0_err.npy')
slip_start = times_df['Start'].to_numpy()

vel_times, __ = get_times_from_error(vel_err, 1, 0.2, window_size=100)
w2_times, __ = get_times_from_error(w2_err, 1, 0.9, window_size=100)

print_measures_from_times(w2_times, vel_times)

'''
eps_range = np.linspace(0.05, 5, 50)
measures_vel = accuracy_measures(slip_start, eps_range, vel_err, verbose=True)
measures_W2 = accuracy_measures(slip_start, eps_range, w2_err, verbose=True)

fig, axs = plt.subplots(2, 1)
plot_accuracy_measures(axs[0], measures_vel, eps_range, 'Wall Velocity')
plot_accuracy_measures(axs[1], measures_W2, eps_range, 'W2B0')
fig.tight_layout()
plt.show()
'''
