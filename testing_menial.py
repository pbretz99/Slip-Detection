
import numpy as np
import matplotlib.pyplot as plt

def threshold(arr, t):
     mask = (arr > t)
     ind_arr = np.indices(arr.shape)
     return arr[mask], np.transpose(ind_arr[:,mask])

arr = np.arange(6).reshape((2, 3))
vals, indices = threshold(arr, 3)
print(arr)
print(vals)
print(indices)
