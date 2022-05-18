
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from Utilities import load_data
from Times import get_times_from_error, get_start_times

pairs = np.array([[0, 2], [5, 10], [6, 7]])
matched = []
for p in pairs:
     if p[0] <= 5:
          matched.append(p.tolist())

for p in pairs.tolist():
     if p in matched:
          print(p)

