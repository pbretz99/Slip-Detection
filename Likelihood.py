'''
Functionality involving Likelihood Functions
'''

# Libraries
import numpy as np

# Local Code
from DLM import filter_sample

# Basic log-likelihood computation
def log_likelihood(innovation, obs_var):

     term_1 = np.sum(np.log(obs_var))
     term_2 = np.sum((innovation) ** 2 / obs_var)

     return -0.5 * (term_1 + term_2)

# Likelihood of data given a switch from M1 to M2 at time t
def switching_likelihood(Model1, Model2, Data, t, init, final, burn_in, reset_to_zero=False):

     if t == init + burn_in:
          results = filter_sample(Model2, Data, init, final)
          ll = log_likelihood(np.array(results.innovation)[burn_in:], np.array(results.obs_var)[burn_in:])
          return np.exp(ll)
     
     results1 = filter_sample(Model1, Data, init, t)
     temp_Model2 = Model2.copy()
     temp_Model2.set_inits(results1)
     results2 = filter_sample(Model2, Data, t, final, reset_to_zero=reset_to_zero)
     innovation = results1.innovation + results2.innovation
     obs_var = results1.obs_var + results2.obs_var
     ll = log_likelihood(np.array(innovation)[burn_in:], np.array(obs_var)[burn_in:])
     return np.exp(ll)

# Get each switching likelihood in range
def full_switching_likelihood(Model1, Model2, Data, init, final, burn_in, reset_to_zero=False):

     likelihoods = []
     for t in range(init+burn_in, final):
          like = switching_likelihood(Model1, Model2, Data, t, init, final, burn_in, reset_to_zero)
          likelihoods.append(like)
     return np.array(likelihoods)
