'''
Model-Based Slip Detection

This code develops a model-based approach to identifying slip
periods in the wall velocity and detecting them in the W2B0 
measure.
'''

# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import set_up_local_discount_filter, set_up_drift_discount_filter, filter_sample
from Plotting import diagnostic_plots, filter_plot, error_plot, colorline
from Times import get_times_from_error, get_accuracy_measures_from_times, get_start_times, print_measures_from_times
from Utilities import load_data, load_data_other_runs

# List of standard models
def get_models():

     ModelVel = set_up_local_discount_filter(0, omega=0.2618, df=0.75, alpha=2, beta=0.0001**2, J=3)
     ModelW2B0 = set_up_drift_discount_filter(0.1, omega=0.5466, df=0.75, alpha=2, beta=0.0001**2, J=3, my_EKF=True)
     ModelW2B1 = set_up_drift_discount_filter(0.1, omega=0.5466, df=0.75, alpha=2, beta=0.0001**2, J=3, my_EKF=True)
     ModelPerc = set_up_drift_discount_filter(0.1, omega=0.5466, df=0.85, alpha=2, beta=0.0001**2, J=3)
     ModelTP0 = set_up_drift_discount_filter(0.1, omega=0.2618, df=0.7, alpha=2, beta=0.0001**2, J=3)

     return [ModelVel, ModelW2B0, ModelW2B1, ModelPerc, ModelTP0]

# Plot Model results on sample
def quick_plot(axs, results, Data, data_label, init, final, kind='filter', burn_in=0):

     init = init + burn_in
     filter_plot(axs[0], results.point_estimate(kind=kind)[burn_in:], Data, init, final, data_label, kind=kind)
     error_plot(axs[1], results.standardized_error()[burn_in:], init, final, data_label)

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
     sample = diagnostic_plots(results, init, window, data_label, lags=15, show_plots=show_diagnostic_plots, verbose=verbose)
     return sample

def get_vel_times(ModelVel, threshold_detect=1.5, threshold_start=0.001, other_runs=False):

     # Load data
     if other_runs: Vel = load_data_other_runs('xvelocity', dynamic=True)
     else: Vel = load_data('xvelocity')

     # Create model
     ModelVel = set_up_local_discount_filter(Vel[1], omega=0.2618, df=0.75, alpha=2, beta=0.0001**2, J=3)

     # Run filter
     results_vel = filter_sample(ModelVel, Vel, 1, len(Vel))

     # Baseline of slips for matching
     times_detect, __ = get_times_from_error(results_vel.standardized_error(), 1, threshold_detect, window_size=100)
     times_start = get_start_times(times_detect, Vel, threshold_start, window_size=5)

     return times_start

def run_results(eps_range, measure, Model, times_vel, verbose=True):

     # Load data
     Vel = load_data('xvelocity')
     Data = load_data(measure)

     # Run filter
     results_data = filter_sample(Model, Data, 1, len(Data))

     # Get average velocity at detection for varying threshold
     measures = [[], [], [], [], []]
     for i in range(len(eps_range)):
          times_data, __ = get_times_from_error(results_data.standardized_error(), 1, threshold=eps_range[i], window_size=25)
          f_p, t_p, med = get_accuracy_measures_from_times(times_data, times_vel, cut_off=150)
          if len(times_data) > 0: med_vel = np.median(Vel[times_data])
          else: med_vel = 0
          measures_current = [f_p, t_p, med, len(times_data), med_vel]
          for j in range(5): measures[j].append(measures_current[j])
          if verbose: print('epsilon = %2.3f, fp = %2.3f percent, tp = %2.3f percent, med = %2.1f, count = %i, med vel = %2.5f' %(eps_range[i], f_p * 100, t_p * 100, med, len(times_data), med_vel))
     
     measures.append(eps_range.tolist())
     return measures

def add_endpoints_to_results(results):
     new_results = [[], [], [], [], [], []]
     for j in range(6):
          if j < 2: new_results[j].append(1)
          else: new_results[j].append(results[j][0])
     for j in range(6): new_results[j] = new_results[j] + results[j]
     for j in range(6):
          if j < 2: new_results[j].append(0)
          else: new_results[j].append(results[j][-1])
     return new_results
     
def bound_results(results, bounds):
     (lower, upper) = bounds
     new_results = [[], [], [], [], [], []]
     T = len(results[0])
     for n in range(T):
          if results[-1][n] > lower and results[-1][n] < upper:
               for j in range(6):
                    new_results[j].append(results[j][n])

     return new_results

def get_norm(results_list, ind):
     N = len(results_list)
     min_val, max_val = min(results_list[0][ind]), max(results_list[0][ind])
     for i in range(1, N):
          min_val, max_val = min(min_val, min(results_list[i][ind])), max(max_val, max(results_list[i][ind]))
     return plt.Normalize(min_val, max_val)

def plot_ROC(results_list, data_labels, bounds_list, edge_colors=None, markers=None, add_endpoints=False, linewidth=1):

     # Truncate at lower bounds
     N = len(results_list)
     for i in range(N):
          results_list[i] = bound_results(results_list[i], bounds_list[i])

     # Add endpoints
     if add_endpoints:
          for i in range(N):
               results_list[i] = add_endpoints_to_results(results_list[i])

     color_labels = ['Detection Count', 'Median Velocity', 'Threshold $\epsilon$']
     fig, axs = plt.subplots(1, 3, figsize=(6.25 * 3, 5))
     for j in range(3):
          norm = get_norm(results_list, j+3)
          ax = axs[j]
          ax.plot([0, 1], [0, 1], ls='--', c='lightgray')
          for i in range(N):
               if i == 0: add_colorbar = True
               else: add_colorbar = False
               colorline(ax, results_list[i][0], results_list[i][1], results_list[i][j+3], cmap='seismic', norm=norm, add_colorbar=add_colorbar, cbar_label=color_labels[j], linewidth=linewidth)
               if markers is not None:
                    if edge_colors is not None:
                         ax.scatter(results_list[i][0], results_list[i][1], c=results_list[i][j+3], edgecolor=edge_colors[i], marker=markers[i], cmap='seismic', norm=norm, alpha=0.5, label=data_labels[i])
                    else:
                         ax.scatter(results_list[i][0], results_list[i][1], c=results_list[i][j+3], marker=markers[i], cmap='seismic', norm=norm, alpha=0.5, label=data_labels[i])
          ax.set_xlabel('False Positive Rate')
          ax.set_ylabel('True Positive Rate')
          ax.set_title('ROC Curve (%s)' %color_labels[j])
          ax.axis('equal')
          if N > 1: ax.legend()
     
     fig.tight_layout()
     plt.show()

def plot_slip_counts(results_list, eps_range, data_labels, colors):

     fig, ax = plt.subplots(figsize=(5, 5))
     N = len(results_list)
     for i in range(N): ax.plot(eps_range, results_list[i][3], c=colors[i], label=data_labels[i])
     ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
     ax.set_xlabel('Detection Threshold $\epsilon$')
     ax.set_ylabel('Detection Count')
     ax.set_title('Number of Detections')
     if N > 1: ax.legend()
     plt.show()

def plot_med_vels(results_list, eps_range, data_labels, colors):

     fig, ax = plt.subplots(figsize=(5, 5))
     N = len(results_list)
     for i in range(N): ax.plot(eps_range, results_list[i][4], c=colors[i], label=data_labels[i])
     ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
     ax.set_xlabel('Detection Threshold $\epsilon$')
     ax.set_ylabel('Median Velocity')
     ax.set_title('Median Velocity at Detection')
     if N > 1: ax.legend()
     plt.show()

# Get stick samples
def run_stick_sample(sample_ranges):

     #Load data
     W2 = load_data('w2_b0')
     Vel = load_data('xvelocity')
     Data = [Vel, W2]
     data_labels = ['X Wall Velocity', 'W2']
     colors = ['gray', 'steelblue']

     # Plot
     fig, axs = plt.subplots(len(sample_ranges), len(Data))
     for j in range(len(sample_ranges)):
          (init, final) = sample_ranges[j]
          for i in range(len(Data)):
               axs[j,i].plot(range(init, final), Data[i][init:final], c=colors[i])
               axs[j,i].set_ylabel(data_labels[i])
               axs[j,i].set_title(data_labels[i] + ' Sample ' + str(j+1))

     fig.tight_layout()
     plt.show()

# Run model diagnostics for Velocity and W2B0
def run_Vel_and_W2_diagnostics(print_vel=True, print_W2=True):
     models = get_models()
     ModelVel, ModelW2 = models[0], models[1]
     ranges = [(9150, 9550), (19425, 19825), (30400, 30700)]
     windows = [(9300, 9400), (19500, 19600), (30500, 30600)]
     samples = []
     for i in range(3):
          print('\nRunning sample (%i, %i)' %windows[i])
          if print_vel: print('\nVelocity:')
          sample_vel = run_diagnostic('xvelocity', ModelVel, 'X Wall Velocity', range=ranges[i], window=windows[i], show_plot=False, show_diagnostic_plots=False, verbose=print_vel)
          if print_W2: print('\nW2:')
          sample_W2 = run_diagnostic('w2_b0', ModelW2, 'W2', range=ranges[i], window=windows[i], show_plot=False, show_diagnostic_plots=False, verbose=print_W2)
          samples.append([sample_vel, sample_W2])
     fig, axs = plt.subplots(3, 2)
     data_labels = ['Velocity', 'W2']
     for j in [0, 1]:
          for i in range(3):
               ax = axs[i,j]
               error_plot(axs[i,j], samples[i][j], windows[i][0], windows[i][1])
               axs[i,j].set_title('%s Error Sample %i' %(data_labels[j], i+1))
     fig.tight_layout()
     plt.show()

# Run threshold variation for Velocity and W2B0
def run_Vel_and_W2_var():
     models = get_models()
     ModelVel, ModelW2 = models[0], models[1]
     vel_times = get_vel_times(models[0])
     eps_range = np.linspace(0, 5, 25)
     results_vel, results_W2 = run_results(eps_range, 'xvelocity', ModelVel, vel_times, verbose=False), run_results(eps_range, 'w2_b0', ModelW2, vel_times)
     plot_slip_counts([results_vel], eps_range, ['Velocity'], ['darkblue'])
     plot_med_vels([results_vel], eps_range, ['Velocity'], ['darkblue'])
     plot_slip_counts([results_vel, results_W2], eps_range, ['Velocity', 'W2'], ['darkblue', 'steelblue'])
     plot_med_vels([results_vel, results_W2], eps_range, ['Velocity', 'W2'], ['darkblue', 'steelblue'])
     plot_ROC([results_W2], ['W2'], [(0.4, 5.1)], linewidth=4)

# Run fixed analysis for Velocity and W2B0
def run_Vel_fixed(threshold_detect=1.5, threshold_start=0.001):

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     ModelVel = get_models()[0]

     # Get times
     results = filter_sample(ModelVel, Vel, 1, len(Vel), verbose=True)
     TimesLarge, __ = get_times_from_error(results.standardized_error(), 1, threshold_detect, window_size=25)
     TimesSmall = get_start_times(TimesLarge, Vel, threshold_start, window_size=5)

     # Plot some results
     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

     axs[0].hist(Vel[TimesLarge], bins=30, color='lightblue', edgecolor='black')
     axs[0].set_xlabel('X Wall Velocity')
     axs[0].set_title('Wall Velocity at Detection')

     diffs = TimesLarge-TimesSmall
     axs[1].hist(diffs[diffs < 150], bins=30, color='lightblue', edgecolor='black')
     axs[1].set_xlabel('Difference')
     axs[1].set_title('Difference Between Detection Time and Start Time')

     fig.tight_layout()
     plt.show()

def run_W2_fixed(threshold_W2=1):
     
     models = get_models()
     ModelVel, ModelW2 = models[0], models[1]

     type = ['Train', 'Test']
     times_W2, times_vel, Vel_list = [], [], []

     for i in [0, 1]:

          print('\nRunning %s\n' %type[i])
          
          if type[i] == 'Train':
               times_vel.append(get_vel_times(ModelVel))
               W2 = load_data('w2_b0')
               Vel_list.append(load_data('xvelocity'))
          else:
               times_vel.append(get_vel_times(ModelVel, other_runs=True))
               W2 = load_data_other_runs('w2_b0', dynamic=True)
               Vel_list.append(load_data_other_runs('xvelocity', dynamic=True))
          
          results = filter_sample(ModelW2, W2, 1, len(W2))
          times_W2.append(get_times_from_error(results.standardized_error(), 1, threshold=threshold_W2, window_size=25)[0])

     for i in [0, 1]:
          print_measures_from_times(times_W2[i], times_vel[i], cut_off=150)
          f_p, t_p, med = get_accuracy_measures_from_times(times_W2[i], times_vel[i], cut_off=150)
          print('\n%s: at threshold %2.5f, fp = %2.3f percent, tp = %2.3f percent, and med = %2.1f\n' %(type[i], threshold_W2, f_p * 100, t_p * 100, med))

     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
     for i in [0, 1]:
          vals = Vel_list[i][times_W2[i]]
          axs[i].hist(vals[vals < 0.003], bins=30, color='lightblue', edgecolor='black')
          axs[i].set_xlabel('X Wall Velocity')
          axs[i].set_title('Wall Velocity at Detection (%s)' %type[i])
          axs[i].set_xlim(right=0.004)
     
     fig.tight_layout()
     plt.show()

# Run threshold variation for other measures
def run_threshold_var_all(eps_range=np.linspace(0, 5, 25)):

     measures = ['xvelocity', 'w2_b0', 'w2_b1', 'percolate_left_right', 'TP0']
     models = get_models()
     data_labels = ['Velocity', 'W2B0', 'W2B1', 'Percolation', 'TPO']
     markers = ['o', '*', 'D', '<']
     colors = ['darkblue', 'steelblue', 'lightblue', 'green', 'red']
     bounds_list = [(0.4, 5.1), (0.2, 3.5), (0.1, 5.1), (0.2, 2.5)]

     vel_times = get_vel_times(models[0])

     results_list = []
     for i in range(len(measures)):
          print('\nRunning %s\n' %data_labels[i])
          results = run_results(eps_range, measures[i], models[i], vel_times)
          results_list.append(results)
     
     plot_slip_counts(results_list, eps_range, data_labels, colors)
     plot_med_vels(results_list, eps_range, data_labels, colors)
     plot_ROC(results_list[1:], data_labels[1:], bounds_list=bounds_list, edge_colors=colors[1:], markers=markers)


if __name__ == "__main__":
     
     # Get stick samples
     #run_stick_sample([(9150, 9550), (19425, 19825), (30400, 30700)])
     
     # Run model diagnostics for Velocity and W2B0
     #run_Vel_and_W2_diagnostics()

     # Run threshold variation for Velocity and W2B0
     #run_Vel_and_W2_var()

     # Run fixed analysis for Velocity and W2B0
     #run_Vel_fixed()
     run_W2_fixed()

     # Run threshold variation for other measures
     #run_threshold_var_all()
