
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, sin, cos



for b in np.flip(np.linspace(0, 2, num=101)):
     err = ((2 * b / (2 + b)) ** 4) * (1 + 2 * b)
     print(b, err)
     if err < 0.0001: break
