
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from Plotting import add_lettering, add_times_to_plot, filter_plot
from Times import get_times_from_error

from Utilities import load_data, load_times, load_data_all, run_bounds
from Plotting import add_subplot_axes

def plot_sample_thresh(ax, epsilons, window, data, err, data_label, colors, add_times=True, lettering=None):
     
     init, final = window
     times_list = []
     for eps in epsilons:
          times, __ = get_times_from_error(err[init:final], init, eps, window_size=25, burn_in=0)
          times_list.append(times)
     
     ax.plot(range(init, final), data[init:final], c='gray', lw=0.75)
     if add_times:
          for times, color in zip(times_list, colors):
               add_times_to_plot(ax, init, final, times, c=color, ls='-', lw=1)
     bottom, top = ax.get_ylim()
     ax.set_ylim(top=bottom + 1.75 * (top - bottom))
     ax.set_ylabel(data_label)
     if lettering is not None:
          add_lettering(ax, lettering, 0.05, 0.8)

     ax_new = add_subplot_axes(ax, [0.35, 0.5, 0.55, 0.4])
     ax_new.plot(range(init, final), err[init:final], c='gray', lw=0.75)
     #ax_new.set_ylabel('NME')
     if add_times:
          for times, color in zip(times_list, colors):
               add_times_to_plot(ax_new, init, final, times, c=color, ls='-', lw=1)
     ax_new.set_ylim(-2.5, 2.5)


def plot_samples(window):

     init, final = window
     Vel = load_data('xvelocity')
     Vel_forecast = np.load('vel_forecast.npy')
     W2 = load_data('w2_b0')
     W2_forecast = np.load('w2_b0_forecast.npy')
     
     for Data, Data_forecast, Data_label in [[Vel, Vel_forecast, 'Velocity'], [W2, W2_forecast, 'W2B0']]:
          fig, ax = plt.subplots()
          filter_plot(ax, Data_forecast[init:final], Data, init, final, Data_label, kind='forecast')
          plt.show()

def plot_samples_with_err(t, window):

     Vel = load_data('xvelocity')
     W2 = load_data('percolate_left_right')
     vel_err = np.load('vel_err.npy')
     w2_err = np.load('w2_b0_err.npy')
     

     init, final = t + window[0], t + window[1]
     fig, axs = plt.subplots(2, 2)
          
     ax = axs[0,0]
     ax.plot(range(init, final), Vel[init:final], c='gray')
     ax.set_title('Velocity Sample')
     ax.set_ylabel('Velocity')

     ax = axs[1,0]
     ax.plot(range(init, final), vel_err[init:final], c='gray')
     ax.set_title('Velocity Error Sample')
     ax.set_ylabel('Error')

     ax = axs[0,1]
     ax.plot(range(init, final), W2[init:final], c='steelblue')
     ax.set_title('W2B0 Sample')
     ax.set_ylabel('W2B0')

     ax = axs[1,1]
     ax.plot(range(init, final), w2_err[init:final], c='steelblue')
     ax.set_title('W2B0 Error Sample')
     ax.set_ylabel('Error')

     for i in range(2):
          for j in range(2):
               ax = axs[i,j]
               ax.axvline(x=t, c='black', ls='--', alpha=0.2)
               ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
          
     fig.tight_layout()
     plt.show()

def run_plot_fig_2():
     init, final = 1, 10000
     
     v_x = load_data('xvelocity')
     x_pos = load_data('xposition')
     y_pos = load_data('y_position')
     slip_intervals = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int).to_numpy()[:-1]
     times = slip_intervals[slip_intervals[:,0] <= final, 0]
     
     fig, axs = plt.subplots(3, 1)

     for ax, data, label in zip(axs, [x_pos, y_pos, v_x], ['x', 'y', '$v_x$']):
          ax.plot(data[init:final], lw=0.75)
          ax.scatter(times, data[times], c='red', s=7, alpha=1)
          ax.set_ylabel(label)
     
     for ax, num in zip(axs, ['(a)', '(b)', '(c)']):
          bottom, top = ax.get_ylim()
          ax.text(1000, top - 0.2 * (top - bottom), num)
          
     axs[2].set_xlabel('t')
     fig.tight_layout()
     plt.show()

def run_plot_fig_3(measure='xvelocity', data_label='$v_x$'):
     
     samples = [(300, 1100), (5850, 6300), (9000, 9550)]
     insets = [(700, 800), (6000, 6100), (9200, 9300)]

     data= load_data(measure)
     slip_intervals = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int).to_numpy()[:-1]

     fig, axs = plt.subplots(3, 1)
     for ax, sample, inset in zip(axs, samples, insets):
          init, final = sample
          slip_start = slip_intervals[(slip_intervals[:,0] <= final) & (slip_intervals[:,0] >= init), 0]
          slip_stop = slip_intervals[(slip_intervals[:,1] <= final) & (slip_intervals[:,1] >= init), 1]
          ax.plot(range(init, final), data[init:final])
          ax.scatter(slip_start, data[slip_start], c='red', s=25)
          ax.scatter(slip_stop, data[slip_stop], c='green', s=25)
          init, final = inset
          ax_new = add_subplot_axes(ax, [0.2, 0.5, 0.5, 0.4])
          ax_new.plot(range(init, final), data[init:final])
          ax.set_ylabel(data_label)
     
     for ax, num in zip(axs, ['(a)', '(b)', '(c)']):
          add_lettering(ax, num, 0.8, 0.7)
     
     axs[2].set_xlabel('t')
     plt.show()

def run_test(file_label, data_label):

     samples = []
     runs = [1, 2, 3, 4]
     for run in runs:
          init, final = run_bounds(run)
          shift = 100000
          samples.append([init+shift, init+shift+1000])
     
     v_x = load_data_all('xvelocity')
     data = load_data_all(file_label)
     slip_intervals = load_times()

     fig, axs = plt.subplots(len(samples), 2)
     for j, current_data, current_label in zip([0, 1], [v_x, data], ['$v_x$', data_label]):
          for ax, sample, num in zip(axs[:,j], samples, runs):
               init, final = sample
               slip_start = slip_intervals[(slip_intervals[:,0] <= final) & (slip_intervals[:,0] >= init), 0]
               slip_stop = slip_intervals[(slip_intervals[:,1] <= final) & (slip_intervals[:,1] >= init), 1]
               ax.plot(range(init, final), current_data[init:final], lw=0.7)
               ax.scatter(slip_start, current_data[slip_start], c='red', s=25)
               ax.scatter(slip_stop, current_data[slip_stop], c='green', s=25)
               ax.set_ylabel(current_label)
               if j == 0:
                    add_lettering(ax, f'Run S1{num}', 0.1, 0.8)
     
     for j in [0, 1]:
          axs[len(samples)-1,j].set_xlabel('t')
     plt.show()

if __name__ == '__main__':

     run_test(file_label='percolate_left_right', data_label='Perc')
     #run_plot_fig_3(measure='w2_b0', data_label='W2B0')
     #plot_samples_with_err(6050, (-200, 150))
     print('Done!')
