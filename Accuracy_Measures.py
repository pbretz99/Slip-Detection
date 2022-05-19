

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


def overlapping(interval_1, interval_2):
     for first, second in [[interval_1, interval_2], [interval_2, interval_1]]:
          for time in first:
               if second[0] <= time <= second[1]:
                    return True
     return False

# Note: inputs need to be lists of pairs (which are themselves lists)
def split_by_overlap(detection_pairs, basis_pairs):
     overlapping_detection, overlapping_basis = [], []
     for detection_interval in detection_pairs:
          for slip_interval in basis_pairs:
               if overlapping(detection_interval, slip_interval):
                    overlapping_detection.append(detection_interval)
                    overlapping_basis.append(slip_interval)
     distinct_detection = []
     for detection_interval in detection_pairs:
          if detection_interval not in overlapping_detection:
               distinct_detection.append(detection_interval)
     distinct_basis = []
     for slip_interval in basis_pairs:
          if slip_interval not in overlapping_basis:
               distinct_basis.append(slip_interval)
     return [overlapping_detection, overlapping_basis], [distinct_detection, distinct_basis]

def split_by_advance(overlapping_detection, overlapping_basis):
     advance_detection, advance_basis = [], []
     for detection_interval, slip_interval in zip(overlapping_detection, overlapping_basis):
          if detection_interval[0] <= slip_interval[0]:
               if detection_interval not in advance_detection:
                    advance_detection.append(detection_interval)
               if slip_interval not in advance_basis:
                    advance_basis.append(slip_interval)
     return advance_detection, advance_basis

def my_measures_overlap(detection_pairs, basis_pairs):
     overlap_list, distinct_list = split_by_overlap(detection_pairs, basis_pairs)
     advance_detection, advance_basis = split_by_advance(overlap_list[0], overlap_list[1])
     N_detections = len(detection_pairs)
     N_slips = len(basis_pairs)
     N_matched = N_slips - len(distinct_list[1])
     N_matched_advance = len(advance_basis)
     f_p = 1 - N_matched / N_detections
     t_p_total = N_matched / N_slips
     t_p_partial = N_matched_advance / N_slips
     return np.array([f_p, t_p_total, t_p_partial])

def pair_times(start, stop):
     if len(start) == 0 or len(stop) == 0:
          return np.array([])
     if stop[0] <= start[0]:
          stop = stop[1:]
     N = min(len(start), len(stop))
     start, stop = start[0:N], stop[0:N]
     return np.stack((start, stop), axis=-1)

def accuracy_measures_overlap(basis_pairs, eps_range, err, verbose=False):
     measures = np.zeros((len(eps_range), 3))
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, stop = get_times_from_error(err, 1, eps_range[i], window_size=25)
          detection_pairs = pair_times(start, stop)
          measures[i] = my_measures_overlap(detection_pairs.tolist(), basis_pairs.tolist()[:-1])
     return measures

def print_measures(measures, eps_range, data_label):
     print(f'\nAccuracy measures for {data_label} with eps ranging from {eps_range[0]} to {eps_range[-1]}:')
     for current_measures, eps in zip(measures.tolist(), eps_range):
          f_p, t_p_total, t_p_partial = current_measures
          print(f'f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}, eps = {eps}')

def accuracy_measures(slip_start, eps_range, err, verbose=False):
     measures = np.zeros((len(eps_range), 3))
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          detection, __ = get_times_from_error(err, 1, eps_range[i], window_size=25)
          measures[i] = my_measures(detection, slip_start)
     return measures

def plot_accuracy_measures(ax, measures, eps_range, data_label, legend=True):
     measure_labels = ['$f_p$', '$t_p$ (total)', '$t_p$ (partial)']
     measure_colors = ['orange', 'steelblue', 'steelblue']
     measure_ls = ['-', '--', '-']
     for i in range(3):
          ax.plot(eps_range, measures[:,i], label=measure_labels[i], c=measure_colors[i], ls=measure_ls[i])
     ax.axhline(y=1, c='lightgray', ls='--')
     ax.set_ylim(bottom=0)
     if legend:
          ax.legend()
     ax.set_xlabel('Error Threshold $\epsilon$')
     ax.set_ylabel('Rate')
     ax.set_title(f'Accuracy Measures for {data_label} Detections')

def plot_accuracy_measures2(ax, measures, eps_range, data_label, legend=True):
     measure_labels = ['$f_p$', '$t_p$']
     measure_colors = ['orange', 'steelblue']
     measure_ls = ['-', '-']
     for i in range(2):
          ax.plot(eps_range, measures[:,i], label=measure_labels[i], c=measure_colors[i], ls=measure_ls[i])
     ax.axhline(y=1, c='lightgray', ls='--')
     ax.set_ylim(bottom=0)
     if legend:
          ax.legend()
     ax.set_xlabel('Error Threshold $\epsilon$')
     ax.set_ylabel('Rate')
     ax.set_title(f'Accuracy Measures for {data_label} Detections')

if __name__ == '__main__':

     times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int)

     #Vel = load_data('xvelocity')
     vel_err = np.load('vel_err.npy')
     w2_err = np.load('w2_b0_err.npy')
     slip_start = times_df['Start'].to_numpy()

     eps_vel, eps_w2 = 0.2, 0.4
     vel_start, vel_stop = get_times_from_error(vel_err, 1, eps_vel, window_size=25)
     w2_start, w2_stop = get_times_from_error(w2_err, 1, eps_w2, window_size=25)

     print(f'At the Velocity eps = {eps_vel} and W2B0 eps = {eps_w2} level:')
     f_p, t_p_total, t_p_partial = my_measures_overlap(pair_times(w2_start, w2_stop).tolist(), pair_times(vel_start, vel_stop).tolist())
     print(f'  f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}')

     '''
     eps_range = np.linspace(0.2, 5, 51)
     #measures_vel = accuracy_measures(slip_start, eps_range, vel_err, verbose=True)
     #measures_W2 = accuracy_measures(slip_start, eps_range, w2_err, verbose=True)
     measures_vel = accuracy_measures_overlap(times_df.to_numpy(), eps_range, vel_err, verbose=True)
     measures_W2 = accuracy_measures_overlap(times_df.to_numpy(), eps_range, w2_err, verbose=True)

     for measures, data_label in [[measures_vel, 'Velocity'], [measures_W2, 'W2B0']]:
          print_measures(measures, eps_range, data_label)

     fig, ax = plt.subplots()
     plot_accuracy_measures(ax, measures_vel, eps_range, 'Wall Velocity')
     plt.show()

     fig, axs = plt.subplots(2, 1)
     plot_accuracy_measures(axs[0], measures_vel, eps_range, 'Wall Velocity')
     plot_accuracy_measures(axs[1], measures_W2, eps_range, 'W2B0', legend=False)
     fig.tight_layout()
     plt.show()'''

