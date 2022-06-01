
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DLM import filter_sample
from Paper_1 import get_models
from Plot_Samples import plot_sample_thresh
from Plotting import plot_accuracy_measures, plot_accuracy_measures_all, plot_advance_measures, add_lettering
from Utilities import load_data, print_tracker, overlapping
from Times import get_times_from_error, get_all_measures_from_times, print_measures_from_times

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

def get_advance_notice(overlapping_detection, overlapping_basis):
     notices, examined_basis = [], []
     for detection_interval, slip_interval in zip(overlapping_detection, overlapping_basis):
          # Only do advance notice for first matching
          if slip_interval not in examined_basis:
               notices.append(slip_interval[0] - detection_interval[0])
               examined_basis.append(slip_interval)
     return notices

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
     __, advance_basis = split_by_advance(overlap_list[0], overlap_list[1])
     advance_notice = get_advance_notice(overlap_list[0], overlap_list[1])
     N_detections = len(detection_pairs)
     N_slips = len(basis_pairs)
     N_matched = N_slips - len(distinct_list[1])
     N_matched_advance = len(advance_basis)
     f_p = 1 - N_matched / N_detections
     t_p_total = N_matched / N_slips
     t_p_partial = N_matched_advance / N_slips
     adv_total = 0
     if len(advance_notice) > 0:
          advance_notice = np.array(advance_notice)
          adv_total = np.median(advance_notice)
     return np.array([t_p_total, t_p_partial, f_p, adv_total])

def pair_times(start, stop):
     if len(start) == 0 or len(stop) == 0:
          return np.array([])
     if stop[0] <= start[0]:
          stop = stop[1:]
     N = min(len(start), len(stop))
     start, stop = start[0:N], stop[0:N]
     return np.stack((start, stop), axis=-1)

def accuracy_measures_overlap(basis_pairs, eps_range, err, verbose=False):
     measures = np.zeros((len(eps_range), 4))
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, stop = get_times_from_error(err, 1, eps_range[i], window_size=25)
          detection_pairs = pair_times(start, stop)
          measures[i] = my_measures_overlap(detection_pairs.tolist(), basis_pairs.tolist()[:-1])
     return measures

def print_measures(measures, eps_range, data_label):
     print(f'\nAccuracy measures for {data_label} with eps ranging from {eps_range[0]} to {eps_range[-1]}:')
     for current_measures, eps in zip(measures.tolist(), eps_range):
          t_p_total, t_p_partial, f_p, adv = current_measures
          print(f'f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}, med. adv. {round(adv, 1)}, eps = {eps}')

def get_detection_counts_and_med_vel(eps_range, err, verbose=False):
     v_x = load_data('xvelocity')
     counts = []
     med_vels = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, __ = get_times_from_error(err, 1, eps_range[i], window_size=25)
          counts.append(len(start))
          if len(start) > 0:
               med_vels.append(np.median(v_x[start]))
          else:
               med_vels.append(0)
     return np.array(counts), np.array(med_vels)

def run_vel_accuracy(eps_vel=0.2):
     
     # Get results for varying epsilon
     times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int)
     vel_err = np.load('vel_err.npy')
     eps_range = np.linspace(eps_vel, 5, 51)
     measures_vel = accuracy_measures_overlap(times_df.to_numpy(), eps_range, vel_err, verbose=True)
     
     # Print and display results
     print_measures(measures_vel, eps_range, 'Velocity')
     fig, axs = plt.subplots(1, 2)
     plot_accuracy_measures(axs[0], measures_vel, eps_range, 'Velocity')
     plot_advance_measures(axs[1], measures_vel, eps_range, 'Velocity', partial=False)
     fig.tight_layout()
     plt.show()

# Change for advance notices (Vel in accuracy_measures_overlap function)
def run_vel_and_w2_comparison(eps_vel=0.2, eps_w2=0.4):
     
     vel_err = np.load('vel_err.npy')
     w2_err = np.load('w2_b0_err.npy')

     # At the individual level, comparison
     vel_start, vel_stop = get_times_from_error(vel_err, 1, eps_vel, window_size=25)
     w2_start, w2_stop = get_times_from_error(w2_err, 1, eps_w2, window_size=25)
     print(f'At the individual level with Velocity eps = {eps_vel} and W2B0 eps = {eps_w2} level:')
     t_p_total, t_p_partial, f_p, adv = my_measures_overlap(pair_times(w2_start, w2_stop).tolist(), pair_times(vel_start, vel_stop).tolist())
     print(f'  f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}, med. adv (total) {adv}')

def run_W2_accuracy(eps_w2=0.4):

     # Plot results for each
     fig, axs = plt.subplots(2, 2)
     print_and_plot_accuracy(axs, 'w2_b0', eps_w2, 'W2B0', 'steelblue')
     for i in [0, 1]:
          for j in [0, 1]:
               axs[i,j].legend()
     plt.show()

def print_and_plot_accuracy(axs, measure, eps, data_label, color, verbose=True):

     times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int)
     err = np.load(f'{measure}_err.npy')

     # Vary in reference to standard slips
     eps_range = np.linspace(eps, 5, 51)
     measures = accuracy_measures_overlap(times_df.to_numpy(), eps_range, err, verbose=verbose)
     
     # Print results for each
     print_measures(measures, eps_range, data_label)
     
     # Plot results for each
     plot_accuracy_measures_all(axs, measures, eps_range, data_label, color)

def run_all_measures_accuracy():
     
     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', 'Perc', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     fig, axs = plt.subplots(2, 2)
     for eps, file_label, data_label, color in zip(eps_mins, file_labels, data_labels, colors):
          print(f'\nRunning {data_label}')
          print_and_plot_accuracy(axs, file_label, eps, data_label, color)
     axs[0,0].legend()
     
     add_lettering(axs[0,0], '(a)', 0.1, 0.8)
     add_lettering(axs[0,1], '(b)', 0.8, 0.8)
     add_lettering(axs[1,0], '(c)', 0.1, 0.8)
     add_lettering(axs[1,1], '(d)', 0.1, 0.1)
     
     plt.show()

def run_all_measures_compare():

     times_df = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int).to_numpy()

     eps_mins = [0.1, 0.4, 0.2, 0.1]
     file_labels = ['vel', 'w2_b0', 'w2_b1', 'perc']
     data_labels = ['$v_x$', 'W2B0', 'W2B1', 'Perc']
     colors = ['darkblue', 'steelblue', 'lightblue', 'green']

     advance_notices = []
     for eps, file_label, data_label in zip(eps_mins, file_labels, data_labels):
          print(f'\nRunning {data_label}')
          err = np.load(f'{file_label}_err.npy')
          start, stop = get_times_from_error(err, 1, threshold=eps, window_size=25)
          overlap_list, __ = split_by_overlap(pair_times(start, stop).tolist(), times_df.tolist())
          advance_notices.append(np.array(get_advance_notice(overlap_list[0], overlap_list[1])))
     
     fig, axs = plt.subplots(2, 2)
     n = 0
     for i in [0, 1]:
          for j in [0, 1]:
               vals = advance_notices[n]
               axs[i,j].hist(vals[vals <= 80], facecolor=colors[n], edgecolor='black', bins=30)
               axs[i,j].set_xlim(-5, 80)
               n += 1
     
     for j in [0, 1]:
          axs[1,j].set_xlabel('Advance Notice')

     add_lettering(axs[0,0], '(a)', 0.8, 0.8)
     add_lettering(axs[0,1], '(b)', 0.8, 0.8)
     add_lettering(axs[1,0], '(c)', 0.8, 0.8)
     add_lettering(axs[1,1], '(d)', 0.8, 0.8)
     
     plt.show()

def run_vel_and_w2_accuracy():
     eps_mins = [0.1, 0.4]
     file_labels = ['vel', 'w2_b0']
     data_labels = ['$v_x$', 'W2B0']
     colors = ['darkblue', 'steelblue']

     fig, axs = plt.subplots(2, 2)
     for eps, file_label, data_label, color in zip(eps_mins, file_labels, data_labels, colors):
          print_and_plot_accuracy(axs, file_label, eps, data_label, color)
     for i in [0, 1]:
          for j in [0, 1]:
               axs[i,j].legend()
     plt.show()

def run_W2_and_other_comparison(eps_w2=0.4, eps_w2b1=0.2, eps_perc=0.2, eps_tp0=0.2):

     file_labels = ['w2_b1', 'percolate_left_right', 'TP0']
     models = get_models()[2:]
     data_labels = ['W2B1', 'Percolation', 'TPO']
     
     w2_err = np.load('w2_b0_err.npy')
     w2_start, w2_stop = get_times_from_error(w2_err, 1, eps_w2, window_size=25)

     times_list = []
     for file_label, model, data_label, eps in zip(file_labels, models, data_labels, [eps_w2b1, eps_perc, eps_tp0]):
          data = load_data(file_label)
          print(f'\nFiltering {data_label}:')
          results = filter_sample(model, data, 1, len(data), verbose=True)
          err = results.standardized_error()
          start, stop = get_times_from_error(err, 1, eps, window_size=25)
          times_list.append(pair_times(start, stop))
     
     # At the individual level, comparison
     for times, data_label, eps in zip(times_list, data_labels, [eps_w2b1, eps_perc, eps_tp0]):
          print(f'At the individual level with (basis) W2B0 eps = {eps_w2} and {data_label} eps = {eps} level:')
          f_p, t_p_total, t_p_partial = my_measures_overlap(pair_times(w2_start, w2_stop).tolist(), times.tolist())
          print(f'  f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}')

def run_vel_and_w2_detection_count():
     
     vel_err = np.load('vel_err.npy')
     w2_err = np.load('w2_b0_err.npy')

     eps_range = np.linspace(0, 5, 51)

     vel_counts, __ = get_detection_counts_and_med_vel(eps_range, vel_err)
     w2_counts, __ = get_detection_counts_and_med_vel(eps_range, w2_err)

     fig = plt.figure()
     ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
     ax1 = plt.subplot2grid((2, 2), (0, 1))
     ax2 = plt.subplot2grid((2, 2), (1, 1))

     ax0.plot(eps_range, vel_counts, label='$v_x$', c='darkblue')
     ax0.plot(eps_range, w2_counts, label='W2B0', c='steelblue')
     ax0.set_xlabel('$T_e$')
     ax0.set_ylabel('Detection Count')
     bottom, top = ax0.get_ylim()
     left, right = ax0.get_xlim()
     colors = ['orange', 'green']
     for eps, color, ls in zip([0.4, 1.5], colors, ['--', '-']):
          ax0.axvline(x=eps, c=color, ls=ls, lw=1)
     ax0.text(right - 0.12 * (right - left), top - 0.9 * (top - bottom), '(a)')
     ax0.legend()

     plot_sample_thresh(ax1, 0.4, 1.5, (550, 1050), data=load_data('xvelocity'), err=np.load('vel_err.npy'), data_label='$v_x$', add_times=True, lettering='(b)', colors=colors)
     plot_sample_thresh(ax2, 0.4, 1.5, (550, 1050), data=load_data('w2_b0'), err=np.load('w2_b0_err.npy'), data_label='W2B0', add_times=True, lettering='(c)', colors=colors)
     ax2.set_xlabel('t')
     plt.show()

def run_all_measures_detection_count(plot_med_vel=True):

     old_file_labels = ['xvelocity', 'percolate_left_right', 'w2_b0']
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', 'Perc', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     eps_range = np.linspace(0, 10, 101)
     errors = []
     counts = []
     med_vels = []
     for file_label in file_labels:
          err = np.load(f'{file_label}_err.npy')
          print(f'\nRunning {file_label}')
          count, med_vel = get_detection_counts_and_med_vel(eps_range, err, verbose=True)
          for list, elem in zip([errors, counts, med_vels], [err, count, med_vel]):
               list.append(elem)

     if not plot_med_vel:
          fig = plt.figure()
          ax0 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
          right_axs = []
          for i in range(4):
               right_axs.append(plt.subplot2grid((4, 2), (i, 1)))
     else:
          fig, axs = plt.subplots(1, 2)
          ax0 = axs[0]
     
     for count, label, color in zip(counts, data_labels, colors):
          ax0.plot(eps_range, count, label=label, c=color)
     ax0.set_xlabel('$T_e$')
     ax0.set_ylabel('Detection Count')
     if not plot_med_vel:
          colors = ['orange', 'green']
          for eps, color, ls in zip([0.4, 1.5], colors, ['--', '-']):
               ax0.axvline(x=eps, c=color, ls=ls, lw=1)
     add_lettering(ax0, '(a)', 0.1, 0.1)
     ax0.legend()

     if plot_med_vel:
          ax1 = axs[1]
          for med_vel, label, color in zip(med_vels, data_labels, colors):
               ax1.plot(eps_range, med_vel, label=label, c=color)
          ax1.set_xlabel('$T_e$')
          ax1.set_ylabel('Median $v_x$ at Detection')
          add_lettering(ax1, '(b)', 0.1, 0.9)
     
     if not plot_med_vel:
          colors = ['orange', 'green']
          for ax, filename, err, data_label, letter in zip(right_axs, old_file_labels, errors, data_labels, ['(b)', '(c)', '(d)', '(e)']):
               plot_sample_thresh(ax, 0.4, 1.5, (950, 1050), data=load_data(filename), err=err, data_label=data_label, add_times=True, lettering=letter, colors=colors)
          right_axs[-1].set_xlabel('t')
     
     plt.show()

if __name__ == '__main__':

     #run_vel_and_w2_detection_count()
     run_all_measures_detection_count(plot_med_vel=True)
     #run_vel_and_w2_comparison()
     #run_vel_and_w2_accuracy()
     run_all_measures_accuracy()
     #run_all_measures_compare()
     #run_W2_and_other_comparison()
