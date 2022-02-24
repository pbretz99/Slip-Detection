
# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import DLMPoly, DLMTrig
from Model_Probabilities import quick_train, running_model_prob
from Plotting import add_times_to_plot
from Times import times_from_prob
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

def run_Vel_changepoint_example():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 9000, 9600

     # Train stick and slip models
     Model = set_up_local_discount_filter(0, omega=0.2582, df=0.8, alpha=2, beta=2*0.0001**2, J=2)
     alpha_stick, beta_stick = quick_train(Model, Vel, 9200, 9400, repeat=200)
     alpha_slip, beta_slip = quick_train(Model, Vel, 9900, 10000, repeat=400)
     
     # Initialize models
     Model_stick, Model_slip = Model.copy(), Model.copy()
     Model_stick.alpha, Model_stick.beta = alpha_stick, beta_stick
     Model_slip.alpha, Model_slip.beta = alpha_slip, beta_slip
     
     # Get switching likelihoods
     ave_times = np.array([400, 100])
     P1, P2 = running_model_prob(Model_stick, Model_slip, Vel, init, final, ave_times, burn_in=50, verbose=True, factor=0.01)

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

def run_Vel_changepoint():

     #Load data
     Vel = load_data('xvelocity')

     # Create model
     init, final = 150, 10000

     # Train stick and slip models
     Model = set_up_local_discount_filter(0, omega=0.2582, df=0.8, alpha=2, beta=2*0.0001**2, J=2)
     alpha_stick, beta_stick = quick_train(Model, Vel, 9200, 9400, repeat=200)
     alpha_slip, beta_slip = quick_train(Model, Vel, 9900, 10000, repeat=400)
     
     # Initialize models
     Model_stick, Model_slip = Model.copy(), Model.copy()
     Model_stick.alpha, Model_stick.beta = alpha_stick, beta_stick
     Model_slip.alpha, Model_slip.beta = alpha_slip, beta_slip
     
     # Get switching likelihoods
     P1, __ = running_model_prob(Model_stick, Model_slip, Vel, init, final, burn_in=50, verbose=True, factor=0.01)
     pad = np.ones((init,))
     P1 = np.concatenate((pad, P1))
     np.save('stick_prob.npy', P1)



def run_Vel_times():

     #Load data
     Vel = load_data('xvelocity')
     stick_prob = np.load('stick_prob.npy')

     TimesStart, TimesEnd = times_from_prob(stick_prob)
     
     # Plot
     init, final = 9000, 10000
     fig, ax = plt.subplots(2, 1, figsize=(5, 5))
     
     ax[0].plot(range(init, final), Vel[init:final], c='grey')
     ax[0].set_ylabel('Vel')
     ax[0].set_title('X Wall Velocity')
     add_times_to_plot(ax[0], init, final, TimesStart, c='red', ls='--')
     add_times_to_plot(ax[0], init, final, TimesEnd, c='green', ls='--')

     ax[1].plot(range(init, final), stick_prob[init:final], c='green', label='Stick')
     #ax[1].plot(range(init, final), 1 - stick_prob[init:final], c='red', label='Slip')
     #ax[1].legend()
     ax[1].set_ylabel('Prob.')
     ax[1].set_title('Regime Probability')
     
     fig.tight_layout()
     plt.show()


if __name__ == "__main__":

     run_Vel_changepoint_example()
