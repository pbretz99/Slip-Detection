
import matplotlib.pyplot as plt
import numpy as np

from Paper_1 import load_data, filter_sample, get_models
from Plotting import filter_plot

def amplitude_plot(measure, Model, data_label, data_range=(9150, 9550), plot_range=None, kind='filter', burn_in=1000):

     if plot_range is None: plot_range = data_range

     # Load data
     Data = load_data(measure)
     
     # Run Model
     (init, final) = data_range
     #init = init - burn_in
     results = filter_sample(Model, Data, init-burn_in, final)

     amps = results.amplitude(separate=True)
     
     # Plot
     fig, axs = plt.subplots(2, 1, figsize=(7, 7))
     (plot_init, plot_final) = plot_range

     filter_plot(axs[0],
                 results.point_estimate(kind=kind)[(plot_init - init + burn_in):(plot_final - init + burn_in)],
                 Data, plot_init, plot_final, 'Velocity', kind=kind)

     for i in range(3):
          axs[1].plot(range(plot_init, plot_final),
                      amps[i,(plot_init - init + burn_in):(plot_final - init + burn_in)],
                      label='Harmonic %i' %(i+1))
     
     axs[1].set_ylabel('Harmonic Amplitude')
     axs[1].set_title('Amplitude of Harmonics')
     axs[1].legend()
     
     fig.tight_layout()
     plt.show()

ModelVel = get_models()[0]
amplitude_plot('xvelocity', get_models()[0], 'Velocity', data_range=(9000, 10000), plot_range=(9100, 9150), kind='forecast')
