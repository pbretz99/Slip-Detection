'''
Model-Based Slip Prediction

By using a model-based approach we can predict slips in advance.
'''

# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Code
from DLM import filter_sample
from Paper_1 import get_models
from Utilities import load_data

if __name__ == 'main':

     (init, final) = (9000, 9550)
     Vel = load_data('xvelocity')
     ModelVel = get_models()[0]
     results = filter_sample(ModelVel, Vel, )
