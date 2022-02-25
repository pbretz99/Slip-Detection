'''
Produce Plots
'''

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from scipy.stats import invgamma

# Local Code
from DLM import filter_sample
from Utilities import get_times_in_range

#############
# Plot Data #
#############

# Plot a filtered estimate over sample
def filter_plot(ax, point_est, Data, init, final, data_label):
     ax.plot(range(init, final), Data[init:final], c='gray', alpha=0.75, label=data_label)
     ax.plot(range(init, final), point_est, c='steelblue', label='Filter')
     ax.legend()
     ax.set_ylabel(data_label)
     ax.set_title('Filtered Est. of %s' %data_label)

# Plot filter error
def error_plot(ax, error, init, final, standardized=True):
     ax.axhline(y=0, c='gray')
     ax.plot(range(init, final), error, c='steelblue')
     ax.set_ylabel('Error')
     title = 'Filter Error'
     if standardized: title = title + ' (Standardized)'
     ax.set_title(title)

# Plot sigma point estimate
def sigma_plot(ax, sigma, init, final, log_diff=False, log_scale=False):
     if log_diff:
          ax.plot(range(init+1, final), np.diff(np.log(sigma)), c='steelblue')
          ax.set_ylabel('Est. log$\sigma^2$')
          ax.set_title('Point Estimate of log $\sigma^2$')
     elif log_scale:
          ax.plot(range(init, final), np.log(sigma), c='steelblue')
          ax.set_ylabel('Est. log$\sigma^2$')
          ax.set_title('Point Estimate of log $\sigma^2$')
     else:
          ax.plot(range(init, final), sigma, c='steelblue')
          ax.set_ylabel('Est. $\sigma^2$')
          ax.set_title('Point Estimate of $\sigma^2$')

# Function to add times to plot
def add_times_to_plot(ax, init, final, Times, **kwargs):
     for t in get_times_in_range(init, final, Times):
          ax.axvline(x=t, **kwargs)

####################
# Diagnostic Plots #
####################

# Normal qq plot
def qq_plot_innovation(ax, innovation, window, init, data_label):
     sample = innovation[(window[0]-init):(window[1]-init)]
     sm.qqplot(sample / np.std(sample), ax=ax, line='45')
     ax.set_title('Normal Q-Q Plot for %s Sample' %data_label)

# PACF plot
def pacf_plot_innovation(ax, innovation, window, init, data_label, lags):
     sample = innovation[(window[0]-init):(window[1]-init)]
     plot_pacf(sample / np.std(sample), ax=ax, lags=lags)
     ax.set_title('Partial Autocorrelation for %s Sample' %data_label)

# ACF plot
def acf_plot_innovation(ax, innovation, window, init, data_label, lags):
     sample = innovation[(window[0]-init):(window[1]-init)]
     plot_acf(sample / np.std(sample), ax=ax, lags=lags)
     ax.set_title('Autocorrelation for %s Sample' %data_label)

# Diagnostic plots
def diagnostic_plots(Model, Data, init, final, window, data_label, lags, partial=False):

     results = filter_sample(Model, Data, init, final)
     err = results.standardized_error()
     sample = err[(window[0]-init):(window[1]-init)]

     fig, ax = plt.subplots(figsize=(5, 5))
     error_plot(ax, sample, window[0], window[1], data_label)
     plt.show()

     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
     qq_plot_innovation(axs[0], err, window, init, data_label)
     if partial: pacf_plot_innovation(axs[1], err, window, init, data_label, lags=lags)
     else: acf_plot_innovation(axs[1], err, window, init, data_label, lags=lags)
     fig.tight_layout()
     plt.show()

#############################
# Plot Measures of Accuracy #
#############################

# ROC Plot Frame
def ROC_frame(ax):
     ax.plot([0, 1], [0, 1], c='gray', ls='--')
     ax.set_xlabel('False Positive Rate')
     ax.set_ylabel('True Positive Rate')

######################
# Plot Distributions #
######################

# Plot inverse gamma
def plot_inv_gamma(ax, alpha, beta, scale=5, num=100, **kwargs):
     RV = invgamma(a=alpha, scale=beta)
     if alpha > 1: mean = beta / (alpha - 1)
     else: mean = beta
     x = np.linspace(0, scale * mean, num=num)
     ax.plot(x, RV.pdf(x), **kwargs)
