
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Accuracy_Measures import get_times_from_error, my_measures_overlap, split_by_overlap, pair_times, get_advance_notice
from Plotting import add_lettering, add_subplot_axes
from Utilities import load_data_all, load_times, run_bounds



def subset_times(times, init, final):
     return times[(times[:,0] >= init) & (times[:,0] <= final),:]

def run_all_measures_compare(run=0, plot_inset=False, show_median=True):

     init, final = run_bounds(run)
     if run == 0:
          times = load_times(all_runs=False)
     else:
          times = subset_times(load_times(), init, final)
     
     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     advance_notices = []
     for eps, file_label, data_label in zip(eps_mins, file_labels, data_labels):
          print(f'\nRunning {data_label}')
          err = np.load(f'{file_label}_err.npy')[init:final]
          start, stop = get_times_from_error(err, init, threshold=eps, window_size=25)
          detection_times = pair_times(start, stop)
          overlap_list, __ = split_by_overlap(detection_times.tolist(), times.tolist())
          t_p_total, t_p_partial, f_p, adv = my_measures_overlap(detection_times.tolist(), times.tolist())
          print(f'{data_label} measure, f_p = {round(f_p, 4)}, t_p (total) = {round(t_p_total, 4)}, t_p (partial) = {round(t_p_partial, 4)}, med. adv. {round(adv, 1)}, eps = {eps}')
          advance_notices.append(np.array(get_advance_notice(overlap_list[0], overlap_list[1])))
     
     fig, ax = plt.subplots()
     for vals, color, label in zip(advance_notices, colors, data_labels):
          sns.kdeplot(vals[vals <= 80], ax=ax, c=color, label=label, bw_adjust=0.5)
          if show_median:
               ax.axvline(x=np.median(vals), c=color, ls='--')
     
     ax.set_xlabel('$t_a$')
     ax.set_ylabel('Density')
     ax.legend()

     plt.show()

def run_all_measures_compare_pairwise(run=0):

     init, final = run_bounds(run)
     if run == 0:
          times = load_times(all_runs=False)
     else:
          times = subset_times(load_times(), init, final)
     
     eps_mins = [0.1, 0.1, 0.4]
     file_labels = ['vel', 'perc', 'w2_b0']
     data_labels = ['$v_x$', '$f_{prl}$', 'W2B0']
     colors = ['darkblue', 'green', 'steelblue']

     detection_times_list = [times]
     for eps, file_label, data_label in zip(eps_mins, file_labels, data_labels):
          print(f'\nRunning {data_label}')
          err = np.load(f'{file_label}_err.npy')[init:final]
          start, stop = get_times_from_error(err, init, threshold=eps, window_size=25)
          detection_times_list.append(pair_times(start, stop))
     
     advance_notices = []
     for basis_times, basis_label in zip(detection_times_list[0:3], ['Standard'] + data_labels[0:2]):
          advance_notices_current_level = []
          for compare_times, compare_label in zip(detection_times_list[1:4], data_labels):
               print(f'\nComparing {compare_label} times to a basis of {basis_label} times')
               overlap_list, __ = split_by_overlap(compare_times.tolist(), basis_times.tolist())
               advance_notices_current_level.append(np.array(get_advance_notice(overlap_list[0], overlap_list[1])))
          advance_notices.append(advance_notices_current_level)
     
     fig, axs = plt.subplots(3, 3)
     max_count = 0
     bounds = (-40, 80)
     for i, basis_label in zip(range(3), ['Standard'] + data_labels[0:2]):
          for j, color, compare_label in zip(range(3), colors, data_labels):
               ax = axs[i,j]
               vals = advance_notices[i][j]
               if i > j:
                    ax.axis('off')
               else:
                    ax.hist(vals[(vals >= bounds[0]) & (vals <= bounds[1])], facecolor=color, edgecolor='black', bins=30)
                    max_count = max(max_count, ax.get_ylim()[1])
                    ax.axvline(x=np.median(vals), c='black', ls='--')
                    ax.set_xlim(bounds[0], bounds[1])
                    print(f'\nIn the median, {compare_label} detections are made {np.median(vals)} frames before {basis_label} times')
     
     for j in range(3):
          axs[j,j].set_xlabel('Advance Notice')

     letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
     num = 0
     for i in range(3):
          for j in range(i, 3):
               add_lettering(axs[i,j], letters[num], 0.05, 0.8)
               num += 1
     
     plt.show()

if __name__ == '__main__':
     
     run_all_measures_compare()
     #run_all_measures_compare_pairwise()
     
     print('Done!')
