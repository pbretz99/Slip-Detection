
import numpy as np

w2_err = np.load('w2_b0_err.npy')
print(np.argwhere(np.isnan(w2_err)))
print(w2_err[13515:13520])
w2_err[np.isnan(w2_err)] = 0
print(w2_err[13515:13520])
