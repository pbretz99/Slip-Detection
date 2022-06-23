import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from scipy.stats import invgamma, shapiro

from Paper_1 import get_models
from Plotting import filter_plot, error_plot, diagnostic_plots
from DLM import filter_sample, set_up_drift_discount_filter, set_up_local_discount_filter
from Utilities import load_data


# Plot Model results on sample
def quick_plot(axs, results, Data, data_label, init, final, kind='filter', burn_in=0):

     init = init + burn_in
     filter_plot(axs[0], results.point_estimate(kind=kind)[burn_in:], Data, init, final, data_label, kind=kind)
     error_plot(axs[1], results.standardized_error()[burn_in:], init, final, data_label)

# Get Shapiro-Wilks p-value and Durbin-Watson Statistic from error sample
def diagnostic_stats(err_sample):
     shapiro_test = shapiro(err_sample)
     return shapiro_test.pvalue, sm.stats.acorr_ljungbox(err_sample, lags=[1]).lb_pvalue.iloc[0]

# Get acceptable delta range for a sample
def get_delta_range(file_label, Model, window, burn_in=1000, precision=0.01, lower_bound=0.5, verbose=False):

     def is_acceptable(delta):
          Model.df = delta
          try:
               results = filter_sample(Model, data, init, final)
               SW_stat, LB_stat = diagnostic_stats(results.standardized_error()[burn_in:])
               if SW_stat > 0.05 and LB_stat > 0.05:
                    return True
               return False
          except:
               return False

     # Load data
     data = load_data(file_label)

     # Set parameters
     (init, final) = window
     init = init - burn_in

     # Initialize iteration
     if verbose:
          print(f'Current delta = {lower_bound}')
     intervals = []
     current_lower_bound = lower_bound
     inside_interval = False
     if is_acceptable(lower_bound):
          inside_interval = True
     
     # Iterate over delta
     for delta in np.linspace(lower_bound + precision, 1 - precision, round((1 - lower_bound - 2 * precision) / precision) + 1):
          if verbose:
               text = f'Current delta = {round(delta, 2)}'
               if is_acceptable(delta):
                    text += ' is acceptable'
               else:
                    text += ' is not acceptable'
               print(text)
          if inside_interval and not is_acceptable(delta):
               intervals.append([current_lower_bound, delta])
               inside_interval = False
          elif not inside_interval and is_acceptable(delta):
               current_lower_bound = delta
               inside_interval = True
     
     # Close last interval if necessary
     if inside_interval:
          intervals.append([current_lower_bound, 1])

     # Return intervals sorted by length
     def key_func(interval):
          return interval[1] - interval[0]
     #return sorted(intervals, key=key_func, reverse=True)
     return intervals

def run_diagnostics_all(window, measures=['v_x', 'f_{plr}', 'W2B0']):

     def v_x_wrapper(J): return set_up_local_discount_filter(0, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=J)
     def f_plr_wrapper(J): return set_up_drift_discount_filter(0.1, omega=0.2618, df=0.88, alpha=2, beta=0.0001**2, J=J)
     def W2B0_wrapper(J): return set_up_drift_discount_filter(0.1, omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=J, my_EKF=True)
     
     # Parameters/Models
     Models = {'v_x': f_plr_wrapper, 'f_{plr}': f_plr_wrapper, 'W2B0': W2B0_wrapper}
     file_labels = {'v_x': 'xvelocity', 'f_{plr}': 'percolate_left_right', 'W2B0': 'w2_b0'}
     
     # Iterate over J
     text = ''
     for J in range(1, 6):
          text += f'\n\nFor J = {J}, the delta ranges are:'
          for key in measures:
               intervals = get_delta_range(file_labels[key], Models[key](J), window, lower_bound=0.2)
               if len(intervals) > 0:
                    text += '\n  '
                    for init, final in intervals[:-1]:
                         text += f'({round(init, 2)}, {round(final, 2)}), '
                    init, final = intervals[-1]
                    text += f'({round(init, 2)}, {round(final, 2)}) for {key}'
               else:
                    text += f'\n  None for {key}'
     print(text)


def run_diagnostic(measure, Model, data_label, range=(9150, 9550), window=(9300, 9400), burn_in=1000, show_plot=True, kind='filter', show_diagnostic_plots=True, verbose=True):
     
     # Load data
     Data = load_data(measure)
     
     # Create model
     (init, final) = range
     init = init - burn_in
     results = filter_sample(Model, Data, init, final)
     if show_plot:
          fig, axs = plt.subplots(2, 1, figsize=(7, 7))
          quick_plot(axs, results, Data, data_label, init, final, burn_in=burn_in, kind=kind)
          fig.tight_layout()
          plt.show()

     # Diagnostic plots
     sample = diagnostic_plots(results, init, window, data_label, lags=15, show_plots=show_diagnostic_plots)

     # Print Statistics
     if verbose:
          SW_stat, LB_stat = diagnostic_stats(sample)
          print('Shapiro-Wilks Test p-value: %2.4f' %SW_stat)
          print('Ljung-Box Test p-vaue (lag 1): %2.2f' %LB_stat)
          
     return sample

# Run model diagnostics for Velocity and W2B0
def run_Vel_diagnostics(ranges, windows, print_vel=True, vel_plot=True):

     Vel = load_data('xvelocity')
     
     models = get_models()
     ModelVel = models[0]
     
     samples = []
     for i in range(3):
          print('\nRunning sample (%i, %i)' %windows[i])
          if print_vel: print('\nVelocity:')
          sample_vel = run_diagnostic('xvelocity', ModelVel, 'X Wall Velocity', range=ranges[i], window=windows[i], show_plot=False, show_diagnostic_plots=False, verbose=print_vel)
          samples.append(sample_vel)
     
     if vel_plot:
          fig, axs = plt.subplots(len(windows), 2)
          for i in range(len(windows)):
               (init, final) = windows[i]
               axs[i,0].plot(range(init, final), Vel[init:final], c='gray')
               axs[i,0].set_ylabel('Velocity')
               axs[i,0].set_title('Velocity Sample %i' %(i+1))
               axs[i,0].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
               
               error_plot(axs[i,1], samples[i], init, final, c='gray')
               axs[i,1].set_title('Sample %i Error' %(i+1))
               axs[i,1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
          fig.tight_layout()
          plt.show()

def run_W2_diagnostics(ranges, windows, print_W2=True, W2_plot=True):
     
     W2 = load_data('w2_b0')

     models = get_models()
     ModelW2 = models[1]
     
     samples = []
     for i in range(3):
          print('\nRunning sample (%i, %i)' %windows[i])
          if print_W2: print('\nW2:')
          sample_W2 = run_diagnostic('w2_b0', ModelW2, 'W2', range=ranges[i], window=windows[i], show_plot=False, show_diagnostic_plots=False, verbose=print_W2)
          samples.append(sample_W2)
     
     if W2_plot:
          fig, axs = plt.subplots(len(windows), 2)
          for i in range(len(windows)):
               (init, final) = windows[i]
               axs[i,0].plot(range(init, final), W2[init:final], c='steelblue')
               axs[i,0].set_ylabel('W2B0')
               axs[i,0].set_title('W2B0 Sample %i' %(i+1))
               axs[i,0].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
               
               error_plot(axs[i,1], samples[i], init, final)
               axs[i,1].set_title('Sample %i Error' %(i+1))
               axs[i,1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
          fig.tight_layout()
          plt.show()

if __name__ == "__main__":

     # Run diagnostics on stick samples
     #sample_ranges = [(600, 1000), (5850, 6250), (9150, 9550)]
     #stick_windows = [(775, 850), (6125, 6200), (9300, 9375)]
     #run_Vel_diagnostics(ranges=sample_ranges, windows=stick_windows)
     #run_W2_diagnostics(ranges=sample_ranges, windows=stick_windows)
     for window in [(6000, 6075)]:
          print(f'\n\nRunning on window ({window[0]}, {window[1]})')
          run_diagnostics_all(window, ['v_x'])


