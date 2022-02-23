
# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import DLMPoly, DLMTrig, filter_sample
from Likelihood import full_switching_likelihood
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

def quick_train(Model, Data, init, final, repeat=1):

     alpha0, beta0 = Model.alpha, Model.beta
     for __ in range(repeat):
          Model_temp = Model.copy()
          Model_temp.alpha, Model_temp.beta = alpha0, beta0
          results = filter_sample(Model_temp, Data, init, final)
          alpha0, beta0 = results.alpha[-1], results.beta[-1]
     return alpha0, beta0

def switching_model_prob(Model1, Model2, Data, init, final, burn_in=0):

     likelihoods1 = full_switching_likelihood(Model2, Model1, Data, init, final, burn_in=burn_in, reset_to_zero=True)
     likelihoods2 = full_switching_likelihood(Model1, Model2, Data, init, final, burn_in=burn_in)
     L1, L2 = np.sum(likelihoods1), np.sum(likelihoods2)
     P1 = L1 / (L1 + L2)
     P2 = L2 / (L1 + L2)
     return P1, P2

def running_model_prob(Model1, Model2, Data, init, final, window_size=50, burn_in=0):
     
     Probs1, Probs2 = [], []
     for final_current in range(init, final):
          print('Running ', final_current)
          init_current = final_current - burn_in - window_size
          P1, P2 = switching_model_prob(Model1, Model2, Data, init_current, final_current, burn_in=burn_in)
          Probs1.append(P1)
          Probs2.append(P2)
     
     return np.array(Probs1), np.array(Probs2)


def run_Vel_sample():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 9200, 9500
     Model = set_up_local_discount_filter(Vel[init], omega=0.2582, df=0.8, alpha=2, beta=2*0.0001**2, J=2)
     
     # Plot
     quick_plot(Model, Vel, 'X Wall Velocity', init, final)
     plot_sigma_est(Model, Vel, 'X Wall Velocity', init, final)

def run_Vel_changepoint():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 9800, 10200

     # Train stick and slip models
     Model = set_up_local_discount_filter(0, omega=0.2582, df=0.8, alpha=2, beta=2*0.0001**2, J=2)
     alpha_stick, beta_stick = quick_train(Model, Vel, 9200, 9400, repeat=200)
     alpha_slip, beta_slip = quick_train(Model, Vel, 9900, 10000, repeat=400)
     
     # Initialize models
     Model_stick, Model_slip = Model.copy(), Model.copy()
     Model_stick.alpha, Model_stick.beta = alpha_stick, beta_stick
     Model_slip.alpha, Model_slip.beta = alpha_slip, beta_slip
     
     # Get switching likelihoods
     P1, P2 = running_model_prob(Model_stick, Model_slip, Vel, init, final, burn_in=50)

     # Run regular filter for point est
     results = filter_sample(Model, Vel, init, final)

     # Plot
     fig, ax = plt.subplots(2, 1, figsize=(5, 5))
     
     ax[0].plot(range(init, final), Vel[init:final], c='grey')
     ax[0].set_ylabel('Vel')
     ax[0].set_title('X Wall Velocity')

     ax[1].plot(range(init, final), P1, c='green', label='Stick')
     ax[1].plot(range(init, final), P2, c='red', label='Slip')
     ax[1].legend()
     ax[1].set_ylabel('Prob.')
     ax[1].set_title('Regime Probability')
     
     fig.tight_layout()
     plt.show()

if __name__ == "__main__":

     run_Vel_changepoint()
