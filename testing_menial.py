
import numpy as np
import pandas as pd

from Utilities import load_data_all, times_from_labels, run_bounds

for run in range(5):
     init, final = run_bounds(run)
     print(f'Run {run}, range ({init}, {final})')
