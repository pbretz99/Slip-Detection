import numpy as np

from Model_Probabilities import conditional_prob_mat, switching_bayes_rule

ave_times = np.array([5, 100])
L = np.array([10, 1])
P = np.array([0.1, 0.9])
print(conditional_prob_mat(ave_times))
print(switching_bayes_rule(L, ave_times, P))