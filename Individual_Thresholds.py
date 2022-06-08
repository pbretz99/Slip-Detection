
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Accuracy_Measures import get_times_from_error, my_measures_overlap, split_by_overlap, pair_times, get_advance_notice
from Plotting import add_lettering, add_subplot_axes
from Utilities import load_data_all, load_times, overlapping, run_bounds



def subset_times(times, init, final):
     return times[(times[:,0] >= init) & (times[:,0] <= final),:]



def detection_times_wrapper(eps, file_label, all_runs=True):
     err = np.load(f'{file_label}_err.npy')
     if not all_runs:
          err = err[0:299999]
     start, stop = get_times_from_error(err, 1, threshold=eps, window_size=25)
     detection_times = pair_times(start, stop)
     return detection_times

def get_paired_advance_notice(overlap_list, all_runs):
     base_times = load_times(all_runs=all_runs)
     overlap_base_list = []
     for times in overlap_list:
          overlap_base_list.append(split_by_overlap(times, base_times.tolist())[0])
     
     already_matched_base_times = []
     advance_notices = []
     for times_1, base_times_1 in zip(overlap_base_list[0][0], overlap_base_list[0][1]):
          for times_2, base_times_2 in zip(overlap_base_list[1][0], overlap_base_list[1][1]):
               if base_times_1[0] == base_times_2[0] and base_times_1[0] not in already_matched_base_times:
                    advance_notices.append([base_times_1[0] - times_1[0], base_times_1[0] - times_2[0]])
                    already_matched_base_times.append(base_times_1[0])
     return np.array(advance_notices)

def plot_t_a_scatter(ax, times_dict, file_labels, all_runs=True):

     data_labels = {'vel': '$v_x$',
                    'perc': '$f_{prl}$',
                    'w2_b0': 'W2B0'}
     
     # Overlap measure 0 and measure 1
     advance_notices = get_paired_advance_notice(split_by_overlap(times_dict[file_labels[0]], times_dict[file_labels[1]])[0], all_runs=all_runs)

     # Plot
     mask = (advance_notices[:,0] <= 80) & (advance_notices[:,1] <= 80)
     ax.scatter(advance_notices[mask,0], advance_notices[mask,1], s=4, alpha=0.9)
     ax.set_xlim(0, 80)
     ax.set_ylim(0, 80)
     ax.plot(ax.get_xlim(), ax.get_ylim(), c='gray', ls='--')
     ax.set_xlabel(f'$t_a$ ({data_labels[file_labels[0]]})')
     ax.set_ylabel(f'$t_a$ ({data_labels[file_labels[1]]})')

def run_t_a_density(all_runs=True, show_median=True):

     times = load_times(all_runs=all_runs).tolist()
     
     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     advance_notices = []
     for eps, file_label, data_label in zip(eps_mins, file_labels, data_labels):
          print(f'\nRunning {data_label}')
          detection_times = detection_times_wrapper(eps, file_label, all_runs=all_runs).tolist()
          overlap_list = split_by_overlap(detection_times, times)[0]
          t_p_total, t_p_partial, f_p, adv = my_measures_overlap(detection_times, times)
          print(f'{data_label} measure, f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}, med. adv. {round(adv, 1)}, eps = {eps}')
          advance_notices.append(get_advance_notice(overlap_list[0], overlap_list[1]))
     
     fig, ax = plt.subplots()
     for vals, color, label in zip(advance_notices, colors, data_labels):
          sns.kdeplot(vals[vals <= 80], ax=ax, c=color, label=label, bw_adjust=0.5)
          if show_median:
               ax.axvline(x=np.median(vals), c=color, ls='--')
     
     ax.set_xlim(-10, 80)

     ax.set_xlabel('$t_a$')
     ax.set_ylabel('Density')
     ax.legend()

     plt.show()

def run_t_a_scatter(all_runs=True):

     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     
     times_dict = {}
     for eps, file_label, data_label in zip(eps_mins, file_labels, data_labels):
          print(f'\nRunning {data_label}')
          times_dict[file_label] = detection_times_wrapper(eps, file_label, all_runs=all_runs).tolist()

     # Plot
     scale = 5
     fig, axs = plt.subplots(1, 3, figsize=(scale * 3, 0.8 * scale))
     plot_t_a_scatter(axs[0], times_dict, ['vel', 'w2_b0'], all_runs=all_runs)
     plot_t_a_scatter(axs[1], times_dict, ['vel', 'perc'], all_runs=all_runs)
     plot_t_a_scatter(axs[2], times_dict, ['perc', 'w2_b0'], all_runs=all_runs)

     add_lettering(axs[0], '(a)', 0.8, 0.1)
     add_lettering(axs[1], '(b)', 0.8, 0.1)
     add_lettering(axs[2], '(c)', 0.8, 0.1)
     
     plt.subplots_adjust(bottom=0.2, wspace=0.3)
     plt.show()

if __name__ == '__main__':
     
     run_t_a_density()
     #run_t_a_scatter()
     
     print('Done!')
