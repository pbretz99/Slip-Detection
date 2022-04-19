
import numpy as np
from matplotlib import pyplot as plt
from Plotting import add_times_to_plot, filter_plot
from Times import get_times_from_error

from Utilities import load_data

def plot_sample_thresh_vel(eps_1, eps_2, window):
     
     init, final = window
     Vel = load_data('xvelocity')
     vel_err = np.load('vel_err.npy')
     vel_forecast = np.load('vel_forecast.npy')
     times_1, __ = get_times_from_error(vel_err[init:final], init, eps_1, window_size=100, burn_in=25)
     times_2, __ = get_times_from_error(vel_err[init:final], init, eps_2, window_size=100, burn_in=25)
     for t in times_1: print(t)

     fig, axs = plt.subplots(2, 1)
     ax = axs[0]
     filter_plot(ax, vel_forecast[init:final], Vel, init, final, 'Velocity', kind='forecast')
     #ax.plot(range(init, final), Vel[init:final], c='gray')
     #ax.set_ylabel('Velocity')
     #ax.set_title('Wall Velocity')
     
     ax = axs[1]
     ax.plot(range(init, final), vel_err[init:final], c='gray')
     ax.set_ylabel('Error')
     ax.set_title('Model Error')

     for ax in axs:
          add_times_to_plot(ax, init, final, times_1, c='red', ls='--')
          add_times_to_plot(ax, init, final, times_2, c='orange', ls='--')

     fig.tight_layout()
     plt.show()

def plot_sample_thresh_W2(eps, window):
     
     init, final = window
     W2 = load_data('w2_b0')
     W2_err = np.load('w2_b0_err.npy')
     W2_forecast = np.load('w2_b0_forecast.npy')
     times, __ = get_times_from_error(W2_err[init:final], init, eps, window_size=100, burn_in=5)
     for t in times: print(t)
     
     fig, axs = plt.subplots(2, 1)
     ax = axs[0]
     filter_plot(ax, W2_forecast[init:final], W2, init, final, 'W2B0', kind='forecast')
     
     ax = axs[1]
     ax.plot(range(init, final), W2_err[init:final], c='gray')
     ax.set_ylabel('Error')
     ax.set_title('Model Error')

     for ax in axs:
          add_times_to_plot(ax, init, final, times, c='red', ls='--', alpha=0.5)
     fig.tight_layout()
     plt.show()

#plot_sample_thresh_vel(0.2, 1.5, (500, 1250))
plot_sample_thresh_W2(0.4, (500, 1250))
