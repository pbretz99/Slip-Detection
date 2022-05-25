
import numpy as np
from matplotlib import pyplot as plt
from Plotting import add_times_to_plot, filter_plot
from Times import get_times_from_error

from Utilities import load_data

def plot_sample_thresh_vel(eps_1, eps_2, window, add_times=True):
     
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
     ax.set_ylabel('Normalized Error')
     ax.set_title('Normalized Model Error')

     if add_times:
          for ax in axs:
               add_times_to_plot(ax, init, final, times_1, c='red', ls='--')
               add_times_to_plot(ax, init, final, times_2, c='orange', ls='--')

     fig.tight_layout()
     plt.show()

def plot_sample_thresh_W2(eps, window, add_times=True):
     
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
     ax.set_ylabel('Normalized Error')
     ax.set_title('Normalized Model Error')

     if add_times:
          for ax in axs:
               add_times_to_plot(ax, init, final, times, c='red', ls='--', alpha=0.5)
     
     fig.tight_layout()
     plt.show()

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

#plot_sample_thresh_vel(0.2, 1.5, (500, 1050), add_times=False)
#plot_sample_thresh_W2(0.4, (500, 1050), add_times=False)

#plot_samples_with_err(3966, (-100, 100))
plot_samples_with_err(119498, (-150, 150))
