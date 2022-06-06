
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DLM import filter_sample
from Paper_1 import get_models
from Plot_Samples import plot_sample_thresh
from Plotting import plot_accuracy_measures, plot_accuracy_measures_all, plot_advance_measures, add_lettering
from Utilities import load_data, load_data_all, print_tracker, overlapping, load_times
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

def get_detection_counts(eps_range, err, verbose=False):
     v_x = load_data_all('xvelocity')
     counts = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, __ = get_times_from_error(err, 1, eps_range[i], window_size=25)
          counts.append(len(start))
     return np.array(counts)

def get_med_vel(eps_range, err, verbose=False, alpha=0.1, return_CI=False):
     v_x = load_data_all('xvelocity')
     med_vels = []
     lower, upper = [], []
     q_l = alpha / 2
     q_u = 1 - alpha / 2
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, __ = get_times_from_error(err, 1, eps_range[i], window_size=25)
          if len(start) > 0:
               med_vels.append(np.median(v_x[start]))
               for q, measure in zip([q_l, q_u], [lower, upper]):
                    measure.append(np.quantile(v_x[start], q=q))
          else:
               for measure in [med_vels, lower, upper]:
                    measure.append(0)
     if return_CI:
          return np.array(med_vels), np.array(lower), np.array(upper)
     else:
          return np.array(med_vels)

def print_and_plot_accuracy(axs, file_label, eps, data_label, color, verbose=True, all_runs=True):

     times = load_times(all_runs=all_runs)
     err = np.load(f'{file_label}_err.npy')
     if not all_runs:
          err = err[0:299999]

     # Vary in reference to standard slips
     eps_range = np.linspace(eps, 4, 41)
     measures = accuracy_measures_overlap(times, eps_range, err, verbose=verbose)
     
     # Print results for each
     print_measures(measures, eps_range, data_label)
     
     # Plot results for each
     plot_accuracy_measures_all(axs, measures, eps_range, data_label, color)

def run_all_measures_accuracy(all_runs=True):
     
     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     fig, axs = plt.subplots(2, 2)
     for eps, file_label, data_label, color in zip(eps_mins, file_labels, data_labels, colors):
          print(f'\nRunning {data_label}')
          print_and_plot_accuracy(axs, file_label, eps, data_label, color, all_runs=all_runs)
     axs[0,0].legend()
     
     add_lettering(axs[0,0], '(a)', 0.1, 0.8)
     add_lettering(axs[1,0], '(b)', 0.1, 0.8)
     add_lettering(axs[0,1], '(c)', 0.8, 0.8)
     add_lettering(axs[1,1], '(d)', 0.1, 0.1)
     
     plt.subplots_adjust(wspace=0.375)
     plt.show()

def run_all_measures_detection_count(all_runs=True):

     old_file_labels = ['xvelocity', 'percolate_left_right', 'w2_b0']
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']
     epsilons = [1.5, 0.4, 0.1]
     epsilon_colors = ['red', 'orange', 'yellowgreen']

     eps_range = np.linspace(0, 10, 101)
     errors = []
     counts = []
     for file_label in file_labels:
          err = np.load(f'{file_label}_err.npy')
          if not all_runs:
               err = err[0:299999]
          print(f'\nRunning {file_label}')
          count = get_detection_counts(eps_range, err, verbose=True)
          for list, elem in zip([errors, counts], [err, count]):
               list.append(elem)

     scale = 5
     fig = plt.figure(figsize=(8*scale, 0.8*scale))
     ax0 = plt.subplot2grid((len(file_labels), 9), (0, 0), rowspan=len(file_labels), colspan=3)
     middle_axs = []
     right_axs = []
     for i in range(len(file_labels)):
          middle_axs.append(plt.subplot2grid((len(file_labels), 9), (i, 3), colspan=3))
          right_axs.append(plt.subplot2grid((len(file_labels), 9), (i, 6), colspan=3))
     
     for count, label, color in zip(counts, data_labels, colors):
          ax0.plot(eps_range, count, label=label, c=color)
     ax0.set_xlabel('$T_e$')
     ax0.set_ylabel('#Detected Events')
     for eps, color in zip(epsilons, epsilon_colors):
          ax0.axvline(x=eps, c=color, ls='-', lw=1)
     bottom, top = ax0.get_ylim()
     ax0.set_ylim(top=bottom + 1.05*(top - bottom))
     add_lettering(ax0, '(a)', 0.3, 0.9)
     ax0.legend()

     
     for ax, filename, err, data_label, letter in zip(middle_axs, old_file_labels, errors, data_labels, ['(b)', '(c)', '(d)']):
          plot_sample_thresh(ax, epsilons, (960, 1050), data=load_data(filename), err=err, data_label=data_label, add_times=True, lettering=letter, colors=epsilon_colors)
     
     for ax, filename, err, data_label, letter in zip(right_axs, old_file_labels, errors, data_labels, ['(e)', '(f)', '(g)']):
          plot_sample_thresh(ax, epsilons, (575, 675), data=load_data(filename), err=err, data_label=data_label, add_times=True, lettering=letter, colors=epsilon_colors)
          ax.set_ylabel(None)
     
     for axs_list in [middle_axs, right_axs]:
          axs_list[-1].set_xlabel('t')
          for ax in axs_list[:-1]:
               ax.set_xticks([])
     
     plt.subplots_adjust(wspace=1.5)
     plt.show()

def run_all_measures_med_vel(all_runs=True, show_CI=False):

     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     eps_range = np.linspace(0, 5, 51)
     errors = []
     med_vels = []
     lowers, uppers = [], []
     for file_label in file_labels:
          err = np.load(f'{file_label}_err.npy')
          if not all_runs:
               err = err[0:299999]
          print(f'\nRunning {file_label}')
          med_vel, lower, upper = get_med_vel(eps_range, err, verbose=True, return_CI=True)
          for list, elem in zip([errors, med_vels, lowers, uppers], [err, med_vel, lower, upper]):
               list.append(elem)

     fig, ax = plt.subplots()
     for med_vel, lower, upper, label, color in zip(med_vels, lowers, uppers, data_labels, colors):
          ax.plot(eps_range, med_vel, label=label, c=color)
          if show_CI:
               #for measure in [upper, lower]:
               #     ax.plot(eps_range, measure, c=color, ls='--')
               ax.fill_between(eps_range, lower, upper, facecolor=color, alpha=0.2)
     ax.set_xlabel('$T_e$')
     ax.set_ylabel('Median $v_x$ at Detection')
     ax.legend()

     plt.show()

if __name__ == '__main__':

     #run_all_measures_detection_count(all_runs=False)
     #run_all_measures_med_vel(all_runs=False, show_CI=True)
     run_all_measures_accuracy(all_runs=False)
     #run_all_measures_compare()
     #run_W2_and_other_comparison()
