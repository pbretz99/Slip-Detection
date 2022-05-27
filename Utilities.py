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
def load_data(filename, data_path="C:/Users/Owner/Documents/Research/Slip Analysis/Data/Long_Run/", dynamic=False):
     data = np.load(data_path + filename + '.npy')
     if dynamic: data = data[1:]
     return data

# Wrapper for loading data from the additional runs
def load_data_other_runs(measure, dynamic=False):
     data_path = "C:/Users/Owner/Documents/Research/Slip Analysis/Data/Additional_Runs/s1%i/"
     Data = np.array([])
     for i in range(1, 5):
          current_data = load_data(measure, data_path%i, dynamic)
          Data = np.concatenate((Data, current_data))
     return Data

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

# Return unique elements in list
def unique(list):
     unique_list = []
     for x in list:
          if x not in unique_list:
               unique_list.append(x)
     return unique_list

# Return everything in list1 that is not in list2
def not_in(list1, list2):
     not_in_list = []
     for x in list1:
          if x not in list2:
               not_in_list.append(x)
     return not_in_list

# Extend an interval by R
def extend(interval, R):
     init, final = interval
     return [init-R, final+R]

# Return True if two intervals (extended by R) overlap
def overlapping(interval_1, interval_2, R=0):
     interval_1, interval_2 = extend(interval_1, R), extend(interval_2, R)
     for first, second in [[interval_1, interval_2], [interval_2, interval_1]]:
          for time in first:
               if second[0] <= time <= second[1]:
                    return True
     return False

# Return a list of interior intervals [x, y] is replaced with [x + R, y - R]
def interior_intervals(intervals, R=0):
     new_intervals = []
     for x, y in intervals:
          if x + R <= y - R:
               new_intervals.append([x + R, y - R])
     return new_intervals

# Return a list of right-aligned intervals of set size; [x, y] is replaced with [y - size, y]
def right_intervals(intervals, size=50):
     new_intervals = []
     for x, y in intervals:
          if x <= y - size:
               new_intervals.append([y - size, y])
     return new_intervals
