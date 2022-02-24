
# Libraries
import numpy as np

# Local Code
from DLM import filter_sample
from Likelihood import full_switching_likelihood
from Utilities import print_tracker

# Train the hyperparameters repeatedly on a single sample
def quick_train(Model, Data, init, final, repeat=1):

     alpha0, beta0 = Model.alpha, Model.beta
     for __ in range(repeat):
          Model_temp = Model.copy()
          Model_temp.alpha, Model_temp.beta = alpha0, beta0
          results = filter_sample(Model_temp, Data, init, final)
          alpha0, beta0 = results.alpha[-1], results.beta[-1]
     return alpha0, beta0

# Markov transition probability matrix
def conditional_prob_mat(ave_times):

     stay_probs = 1 - 1 / ave_times
     switch_probs = (1 / ave_times) * (1 / (len(ave_times) - 1))
     mat = np.zeros((len(ave_times), len(ave_times)))
     for i in range(len(ave_times)):
          for j in range(len(ave_times)):
               if i == j:
                    mat[i,j] = stay_probs[i]
               else:
                    mat[i,j] = switch_probs[i]
     return mat

# Apply Baye's rule to a Markovian process
def switching_bayes_rule(likelihoods, ave_times, prev_prob):

     conditional_prob = conditional_prob_mat(ave_times)
     prob = []
     for i in range(len(likelihoods)):
          prior = np.sum(conditional_prob[i] * prev_prob)
          posterior = likelihoods[i] * prior
          prob.append(posterior)
     prob = np.array(prob)
     return prob / np.sum(prob)

# Get model probabilities under the single-switch assumption
def switching_model_prob(Model1, Model2, Data, init, final, ave_times, prev_prob, burn_in=0):

     likelihoods1 = full_switching_likelihood(Model2, Model1, Data, init, final, burn_in=burn_in, reset_to_zero=True)
     likelihoods2 = full_switching_likelihood(Model1, Model2, Data, init, final, burn_in=burn_in)
     L1, L2 = np.sum(likelihoods1), np.sum(likelihoods2)
     probs = switching_bayes_rule(np.array([L1, L2]), ave_times, prev_prob)
     return probs[0], probs[1]

# Get array of running model probabilities
def running_model_prob(Model1, Model2, Data, init, final, ave_times, window_size=50, burn_in=0, verbose=False, factor=0.1):
     
     Probs1, Probs2 = [], []
     prev_prob = [0.5, 0.5]
     for final_current in range(init, final):
          if verbose: print_tracker(final_current-init, final-init, factor=factor)
          init_current = final_current - burn_in - window_size
          P1, P2 = switching_model_prob(Model1, Model2, Data, init_current, final_current, ave_times, np.array(prev_prob), burn_in=burn_in)
          Probs1.append(P1)
          Probs2.append(P2)
          prev_prob = [P1, P2]
     
     if verbose: print('Complete!')
     return np.array(Probs1), np.array(Probs2)
