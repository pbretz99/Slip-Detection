'''
Utility Functions
'''

# Libraries
import numpy as np
from scipy.stats import norm

# Simple progress tracker
def print_tracker(i, N, factor=0.1):
     skip = round(factor * N)
     if i % skip == 0: print(round((i / N) * 100), 'percent')

# Load data stored as a numpy array
def load_data(filename, data_path="C:/Users/Owner/Documents/Research/Slip Analysis/Data/Long_Run/"):
     return np.load(data_path + filename + '.npy')

# Get times in a given range
def get_times_in_range(init, final, Times):
     ret = []
     for t in Times:
          if t >= init and t <= final:
               ret.append(t)
     return np.array(ret)

# Get probability of credible interval at threshold eps
def get_prob_of_credible_interval(eps):
     p_lower = norm.cdf(-eps)
     p = 1 - 2 * p_lower
     return p

# Ensure np shape arrays are all 2-d
def check_shape(vec, column=True):
     ret = vec
     if len(vec.shape) == 1:
          if column: ret = vec.reshape((len(vec), 1))
          else: ret = vec.reshape((1, len(vec)))
     return ret

# Ensure square matrices
def check_square(arr):
     ret = check_shape(arr)
     if ret.shape[0] != ret.shape[1]: print('Error! Matrix is not square.')
     return ret
