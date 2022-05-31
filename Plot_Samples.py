
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from Plotting import add_times_to_plot, filter_plot
from Times import get_times_from_error

from Utilities import load_data
from Plotting import add_subplot_axes

def plot_sample_thresh(ax, eps_1, eps_2, window, data, err, add_times=True, lettering=None, color='steelblue'):
     
     init, final = window
     times_1, __ = get_times_from_error(err[init:final], init, eps_1, window_size=25, burn_in=25)
     times_2, __ = get_times_from_error(err[init:final], init, eps_2, window_size=25, burn_in=25)
     for t in times_1: print(t)

     ax.plot(range(init, final), err[init:final], c='gray', lw=0.75)
     if add_times:
          add_times_to_plot(ax, init, final, times_2, c=color, ls='-', lw=1.5)
          add_times_to_plot(ax, init, final, times_1, c=color, ls='--', lw=1)
     bottom, top = ax.get_ylim()
     ax.set_ylim(bottom, top + 0.75 * (top - bottom))
     ax.set_ylabel('NME')
     if lettering is not None:
          bottom, top = ax.get_ylim()
          left, right = ax.get_xlim()
          ax.text(left + 0.1 * (right - left), top - 0.2 * (top - bottom), lettering)
     

     ax_new = add_subplot_axes(ax, [0.3, 0.5, 0.6, 0.4])
     ax_new.plot(range(init, final), data[init:final], c='gray', lw=0.75)
     #ax_new.set_ylabel('$v_x$')
     if add_times:
          add_times_to_plot(ax_new, init, final, times_2, c=color, ls='-', lw=1.5)
          add_times_to_plot(ax_new, init, final, times_1, c=color, ls='--', lw=1)


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
     W2 = load_data('w2_b0')
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
          bottom, top = ax.get_ylim()
          left, right = ax.get_xlim()
          ax.text(right - 0.2 * (right - left), top - 0.3 * (top - bottom), num)
     
     axs[2].set_xlabel('t')
     plt.show()

if __name__ == '__main__':

     #run_plot_fig_3(measure='w2_b0', data_label='W2B0')

     fig = plt.figure()
     ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
     ax1 = plt.subplot2grid((2, 2), (0, 1))
     ax2 = plt.subplot2grid((2, 2), (1, 1))

     plot_sample_thresh(ax1, 0.2, 1.5, (550, 1050), data=load_data('xvelocity'), err=np.load('vel_err.npy'), add_times=True, lettering='(b)', color='darkblue')
     plot_sample_thresh(ax2, 0.4, 1.75, (550, 1050), data=load_data('w2_b0'), err=np.load('w2_b0_err.npy'), add_times=True, lettering='(c)')
     ax2.set_xlabel('t')
     plt.show()

#plot_samples_with_err(3966, (-100, 100))
#plot_samples_with_err(119498, (-150, 150))
#plot_samples_with_err(13517, (-50, 50))