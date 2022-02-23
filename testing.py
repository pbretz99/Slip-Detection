
# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import DLMPoly, DLMTrig, filter_sample
from Plotting import diagnostic_plots, filter_plot, error_plot, ROC_frame, plot_inv_gamma
from Times import get_times_from_labels, get_all_measures_from_times, clean_times, get_times_from_vel
from Utilities import load_data

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

     results = filter_sample(Model, Data, init, final)
     
     fig, axs = plt.subplots(2, 1, figsize=(8, 8))
     filter_plot(axs[0], results.point_estimate(), Data, init, final, data_label)
     error_plot(axs[1], results.standardized_error(), init, final, data_label)
     fig.tight_layout()
     plt.show()

# Plot sigma estimate
def plot_sigma_est(Model, Data, data_label, init, final):

     results = filter_sample(Model, Data, init, final)

     fig, ax = plt.subplots(figsize=(7,7))
     ax.plot(range(init, final), results.var_point_estimate())
     ax.set_ylim(bottom=0)
     plt.show()

def run_Vel_sample():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 9200, 10000
     Model = set_up_local_discount_filter(Vel[init], omega=0.2582, df=0.8, alpha=2, beta=0.0001**2, J=2)
     
     # Plot
     quick_plot(Model, Vel, 'X Wall Velocity', init, final)
     plot_sigma_est(Model, Vel, 'X Wall Velocity', init, final)

if __name__ == "__main__":

     run_Vel_sample()
