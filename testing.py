import numpy as np

from Paper_1 import get_models, run_diagnostic

run_diagnostic('xvelocity',
               get_models()[0],
               'Velocity',
               range=(9000, 10000),
               show_diagnostic_plots=False,
               kind='forecast')

