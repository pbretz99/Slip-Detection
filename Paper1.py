
# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import DLMPoly, DLMTrig, filter_sample
from Plotting import diagnostic_plots, filter_plot, error_plot, ROC_frame, sigma_plot
from Times import get_times_from_labels, get_all_measures_from_times, clean_times, get_times_from_vel, get_times_from_sigma, print_measures_from_times
from Utilities import load_data, print_tracker

# Initialize discount filter with local level and drift term + periodic
def set_up_drift_discount_filter(init_val, omega, df, alpha, beta, J=2):

     m = np.array([[init_val], [0]])
     C = np.array([[0.05, 0.01], [0.01, 0.05]])

     Model = DLMPoly(m, C, [0, 0], np.array([[0]]))
     ModelTrig = DLMTrig(1, omega, J, 0, np.array([[0]]))
     Model.add_model(ModelTrig)
     
     discount_Model = Model.to_discount(df, alpha, beta)
     
     return discount_Model

# Initialize discount filter with local level + periodic
def set_up_local_discount_filter(init_val, omega, df, alpha, beta, J=2):

     m = np.array([[init_val]])
     C = np.array([[0.05]])

     Model = DLMPoly(m, C, [0], np.array([[0]]))
     ModelTrig = DLMTrig(1, omega, J, 0, np.array([[0]]))
     Model.add_model(ModelTrig)
     
     discount_Model = Model.to_discount(df, alpha, beta)
     
     return discount_Model

# Plot Model results on sample
def quick_plot(axs, results, Data, data_label, init, final, burn_in=0):

     init = init + burn_in
     filter_plot(axs[0], results.point_estimate()[burn_in:], Data, init, final, data_label)
     #error_plot(axs[1], results.standardized_error()[burn_in:], init, final, data_label)
     sigma_plot(axs[1], results.var_point_estimate()[burn_in:], init, final, log_scale=True)

# ROC Curve
def get_ROC(mem_range, Model, Data, TimesLabels, add_endpoints=True, **kwargs):

     f_p, t_p, med = [], [], []

     for Times in TimesLabels:
          
          t_p_array, f_p_array = [], []
          med_array = []

          if add_endpoints:
               t_p_array.append(1)
               f_p_array.append(1)
          
          f_p.append(f_p_array)
          t_p.append(t_p_array)
          med.append(med_array)

     for mem in mem_range:
               
          print('\nRunning mem =', mem)
          TimesData = times_for_mem(Model, Data, mem)

          for i in range(len(TimesLabels)):

               f_p_current, t_p_current, med_current = get_accuracy_measures_from_times(TimesData, TimesLabels[i], **kwargs)
               print('FP = %2.3f percent, TP = %2.3f percent, med = %2.2f' %(f_p_current * 100, t_p_current * 100, med_current))
               t_p[i].append(t_p_current)
               f_p[i].append(f_p_current)
               med[i].append(med_current)
     
     if add_endpoints:
          for i in range(len(TimesLabels)):
               t_p[i].append(0)
               f_p[i].append(0)
     
     return f_p, t_p, med

# Get times for a memory parameter
def times_for_mem(Model, Data, memory):
     results = filter_sample(Model, Data, 0, len(Data), forgetful=True, memory=memory)
     sigma = np.log(results.var_point_estimate())
     TimesData, __ = get_times_from_sigma(sigma, 0, window_start=[-25, 10], window_end=[-10, 50], burn_in=100)
     return TimesData

# Wrapper function to get false positive, true positive, and median
def get_accuracy_measures_from_times(TimesData, Times, cut_off=100, pad=25):

     f_n_dict, f_p, advance_dict, __ = get_all_measures_from_times(TimesData, Times, cut_off=100, pad=pad)
     t_p = 1 - f_n_dict['Total']
     med = advance_dict['Median']
     return f_p, t_p, med

def run_stick_sample():

     #Load data
     W2 = load_data('w2_b0')
     Vel = load_data('xvelocity')

     # Plot
     init, final = 9200, 9300
     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
     
     axs[0].plot(range(init, final), Vel[init:final], c='gray')
     axs[0].set_ylabel('X Wall Velocity')
     axs[0].set_title('X Wall Velocity Sample')

     axs[1].plot(range(init, final), W2[init:final], c='steelblue')
     axs[1].set_ylabel('W2')
     axs[1].set_title('W2 Sample')

     fig.tight_layout()
     plt.show()


def run_W2_diagnostic():
     
     # Load data
     W2 = load_data('w2_b0')
     
     # Create model
     init, final = 8900, 10000
     ModelW2 = set_up_drift_discount_filter(W2[init], omega=0.5466, df=0.8, alpha=2, beta=0.5**2, J=3)

     # Plot
     results = filter_sample(ModelW2, W2, init, final, forgetful=True, memory=50)
     fig, axs = plt.subplots(3, 1, figsize=(7, 7))
     quick_plot(axs, results, W2, 'W2', init, final, burn_in=100)
     fig.tight_layout()
     plt.show()

     # Diagnostic plots
     diagnostic_plots(results, W2, 9000, 9500, [9220, 9320], 'W2', lags=15)

def run_Vel_diagnostic():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 8900, 9600
     ModelVel = set_up_drift_discount_filter(Vel[init], omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=3)
     memories = [50, 150]

     fig, axs = plt.subplots(2, 2, figsize=(14, 7))
     results = []
     for i in range(2):

          # Plot
          results_current = filter_sample(ModelVel, Vel, init, final, forgetful=True, memory=memories[i])
          results.append(results_current)
          filter_plot(axs[0,i], results_current.point_estimate()[100:], Vel, init+100, final, 'X Wall Velocity')
          axs[0,i].set_title(axs[0,i].get_title() + '\n$N_{mem}$ = %i' %memories[i])
          sigma_plot(axs[1,i], results_current.var_point_estimate()[100:], init+100, final, log_scale=True)
          i += 1
     fig.tight_layout()
     plt.show()

     # Diagnostic plots
     for i in range(2):
          print('\nMemory = %i' %memories[i])
          diagnostic_plots(results[i], Vel, 9000, 9500, [9220, 9320], 'X Wall Velocity', lags=15)


def run_Vel_times():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     ModelVel = set_up_drift_discount_filter(Vel[1], omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=3)

     # Get counts of slips for varying memory
     num_slips = []
     mem_range = np.linspace(1, 750, num=200)
     for i in range(len(mem_range)):
          print_tracker(i, len(mem_range), factor=0.01)
          results = filter_sample(ModelVel, Vel, 1, len(Vel), forgetful=True, memory=mem_range[i])
          slip_start, __ = get_times_from_sigma(np.log(results.var_point_estimate()), 1, window_start=[-25, 10], window_end=[-10, 50], burn_in=100)
          num_slips.append(len(slip_start))
     print('Complete!')

     # Plot results
     c = 1
     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
     ax = axs[0]
     ax.plot(mem_range[c:], num_slips[c:], c='steelblue')
     ax.set_ylim(bottom=0)
     ax.set_xlabel('$N_{mem}$')
     ax.set_ylabel('Slip Count')
     ax.set_title('Slip Count by $N_{mem}$')
     
     ax = axs[1]
     ax.plot(mem_range[(c+1):], np.diff(num_slips[c:]), c='steelblue')
     ax.set_xlabel('$N_{mem}$')
     ax.set_ylabel('Differenced Slip Count')
     ax.set_title('Differenced Slip Count by $N_{mem}$')

     fig.tight_layout()
     plt.show()

def run_ROC_comparison():

     #Load data
     W2 = load_data('w2_b0')
     Vel = load_data('xvelocity')

     # Create model
     ModelW2 = set_up_drift_discount_filter(W2[1], omega=0.5466, df=0.8, alpha=2, beta=0.5**2, J=3)
     ModelVel = set_up_drift_discount_filter(Vel[1], omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=3)

     # Get times
     TimesLabels = get_times_from_labels(load_data('labels_stick_slip'))
     print('Running micro-slips')
     TimesVel_micro = times_for_mem(ModelVel, Vel[1:], memory=50)
     print('Running regular slips')
     TimesVel_large = times_for_mem(ModelVel, Vel[1:], memory=150)

     # Compare ROC
     N = 10
     mem_range = np.linspace(5, 350, num=N)
     print('Running W2')
     f_p, t_p, __ = get_ROC(mem_range, ModelW2.copy(), W2[1:], [TimesVel_micro, TimesVel_large, TimesLabels], pad=25, add_endpoints=False)
     
     # ROC plots
     '''
     fig, ax = plt.subplots(figsize=(5, 5))
     ROC_frame(ax)
     #ax.plot(f_p[0], t_p[0], c='black', ls='--', label='Label Times')
     ax.plot(f_p[0], t_p[0], c='steelblue', label='Wall Vel $N_{mem}$ = 50')
     ax.plot(f_p[1], t_p[1], c='darkblue', label='Wall Vel $N_{mem}$ = 150')
     ax.legend()
     plt.show()
     '''

     # Direct plots
     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
     ax = axs[0]
     ax.plot(mem_range, f_p[0], c='steelblue', ls='--', label='FP, $N_{mem}=50$')
     ax.plot(mem_range, f_p[1], c='darkblue', ls='--', label='FP, $N_{mem}=150$')
     ax.plot(mem_range, f_p[2], c='green', ls='--', label='FP, Labels')
     ax.set_ylim(0, 1)
     ax.set_xlabel('$N_{mem}$')
     ax.legend()

     ax = axs[1]
     ax.plot(mem_range, t_p[0], c='steelblue', label='TP, $N_{mem}=50$')
     ax.plot(mem_range, t_p[1], c='darkblue', label='TP, $N_{mem}=150$')
     ax.plot(mem_range, t_p[2], c='green', label='TP, Labels')
     ax.set_ylim(0, 1)
     ax.set_xlabel('$N_{mem}$')
     ax.legend()

     fig.tight_layout()
     plt.show()

def run_particular_mem_pair(mem_W2, mem_vel=50, **kwargs):

     #Load data
     W2 = load_data('w2_b0')
     Vel = load_data('xvelocity')

     # Create model
     ModelW2 = set_up_drift_discount_filter(W2[1], omega=0.5466, df=0.8, alpha=2, beta=0.5**2, J=3)
     ModelVel = set_up_drift_discount_filter(Vel[1], omega=0.2618, df=0.8, alpha=2, beta=0.0001**2, J=3)

     # Get times
     print('Running Vel')
     TimesVel = times_for_mem(ModelVel, Vel[1:], memory=mem_vel)
     print('Running W2')
     TimesW2 = times_for_mem(ModelW2, W2[1:], memory=mem_W2)

     # Print results
     print_measures_from_times(TimesW2, TimesVel, **kwargs)

if __name__ == "__main__":

     #run_stick_sample()
     #run_W2_diagnostic()
     #run_Vel_diagnostic()
     #run_Vel_times()
     #run_ROC_comparison()
     run_particular_mem_pair(80, 150, pad=5)
