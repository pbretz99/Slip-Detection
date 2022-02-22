
# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import DLMPoly, DLMTrig, filter_sample
from Plotting import diagnostic_plots, filter_plot, error_plot, ROC_frame
from Times import get_times_from_labels, get_all_measures_from_times, clean_times, get_times_from_vel
from Utilities import load_data

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
def quick_plot(Model, Data, data_label, init, final):

     point_est, innovation, __ = filter_sample(Model, Data, init, final)

     fig, axs = plt.subplots(2, 1, figsize=(8, 8))
     filter_plot(axs[0], point_est, Data, init, final, data_label)
     error_plot(axs[1], innovation, init, final, data_label)
     fig.tight_layout()
     plt.show()

# ROC Curve
def get_ROC(eps_range, Model, Data, TimesLabels, add_endpoints=True, pad=25, **kwargs):

     __, innovation, obs_var = filter_sample(Model, Data, 0, len(Data))
     err = innovation / np.sqrt(obs_var)

     f_p, t_p, med = [], [], []

     for Times in TimesLabels:
          
          t_p_array, f_p_array = [], []
          med_array = []

          if add_endpoints:
               t_p_array.append(1)
               f_p_array.append(1)

          for eps in eps_range:
          
               TimesData_raw = get_times_from_vel(err, 0, threshold=eps, burn_in=100)[0]
               TimesData = clean_times(TimesData_raw, Data, **kwargs)

               f_n_dict, f_p_current, advance_dict, __ = get_all_measures_from_times(TimesData, Times, cut_off=100, pad=pad)
               t_p_array.append(1 - f_n_dict['Total'])
               f_p_array.append(f_p_current)
               med_array.append(advance_dict['Median'])
     
          if add_endpoints:
               t_p_array.append(0)
               f_p_array.append(0)
          
          f_p.append(f_p_array)
          t_p.append(t_p_array)
          med.append(med_array)
     
     return f_p, t_p, med

# Get times for a threshold
def times_for_thresh(Model, Data, threshold, **kwargs):
     __, innovation, obs_var = filter_sample(Model.copy(), Data, 0, len(Data))
     err = innovation / np.sqrt(obs_var)
     TimesData_raw = get_times_from_vel(err, 0, threshold=threshold, burn_in=100)[0]
     TimesData = clean_times(TimesData_raw, Data, **kwargs)
     return TimesData

def run_W2_diagnostic():
     
     # Load data
     W2 = load_data('w2_b0')
     
     # Create model
     init, final = 9000, 10000
     Model = set_up_drift_discount_filter(np.log(W2[init]), omega=0.5466, df=0.9, alpha=2, beta=0.5**2, J=2)
     
     # Plot
     quick_plot(Model, np.log(W2), 'Log W2', init, final)

     # Diagnostic plots
     diagnostic_plots(Model, np.log(W2), 9000, 9500, [9200, 9300], 'Log W2', lags=15)

def run_Vel_diagnostic():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 9000, 10000
     Model = set_up_local_discount_filter(Vel[init], omega=0.2582, df=0.8, alpha=2, beta=0.0001**2, J=2)
     
     # Plot
     quick_plot(Model, Vel, 'X Wall Velocity', init, final)

     # Diagnostic plots
     diagnostic_plots(Model, Vel, 9100, 9500, [9200, 9300], 'X Wall Velocity', lags=25)

def run_ROC_comparison():

     #Load data
     W2 = load_data('w2_b0')
     Vel = load_data('xvelocity')

     # Create model
     ModelW2 = set_up_drift_discount_filter(np.log(W2[1]), omega=0.5466, df=0.9, alpha=2, beta=0.5**2, J=2)
     ModelVel = set_up_local_discount_filter(Vel[1], omega=0.2582, df=0.8, alpha=2, beta=0.0001**2, J=2)

     # Get times
     TimesLabels = get_times_from_labels(load_data('labels_stick_slip'))
     TimesVel = times_for_thresh(ModelVel, Vel[1:], threshold=1.3, data_max=0.01, dist_max=200)

     # Compare ROC
     N_eps = 15
     eps_range = np.linspace(0, 8, num=N_eps)
     f_p, t_p, __ = get_ROC(eps_range, ModelW2.copy(), np.log(W2[1:]), [TimesLabels, TimesVel], pad=25, data_max=np.log(0.02), dist_max=50)
     
     # ROC plots
     fig, ax = plt.subplots(figsize=(5, 5))
     ROC_frame(ax)
     ax.plot(f_p[0], t_p[0], c='black', ls='--', label='Label Times')
     ax.plot(f_p[1], t_p[1], c='steelblue', label='Wall Vel Times')
     ax.legend()
     plt.show()

if __name__ == "__main__":

     run_W2_diagnostic()
     run_Vel_diagnostic()
     run_ROC_comparison()
