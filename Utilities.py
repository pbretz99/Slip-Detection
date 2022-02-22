'''
Utility Functions
'''

# Libraries
import numpy as np

# Load data stored as a numpy array
def load_data(filename, data_path="C:/Users/Owner/Documents/Research/Slip Analysis/Data/Long_Run/"):
     return np.load(data_path + filename + '.npy')

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
