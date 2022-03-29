'''
Produce Plots
'''

# Libraries
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.cm as cm
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import invgamma, shapiro

# Local Code
from DLM import filter_sample
from Utilities import get_times_in_range

#############
# Plot Data #
#############

# Plot a filtered estimate over sample
def filter_plot(ax, point_est, Data, init, final, data_label, kind='filter'):
     if kind == 'filter': kind_label = 'Filter'
     elif kind == 'forecast': kind_label = 'Forecast'
     elif kind == 'level': kind_label = 'Local Level'
     else:
          print('Invalid kind. Valid kinds are filter and forecast.')
          return 0
     ax.plot(range(init, final), Data[init:final], c='gray', alpha=0.75, label=data_label)
     ax.plot(range(init, final), point_est, c='steelblue', label=kind_label)
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
def diagnostic_plots(results, init, window, data_label, lags, partial=False, verbose=True, show_plots=True):

     err = results.standardized_error()
     sample = err[(window[0]-init):(window[1]-init)]

     # Print Statistics
     if verbose:
          shapiro_test = shapiro(sample)
          print('Shapiro-Wilks Test p-value: %2.4f' %shapiro_test.pvalue)
          print('Durbin-Watson Statistic: %2.2f' %durbin_watson(sample))

     if show_plots:
          fig, ax = plt.subplots(figsize=(5, 5))
          error_plot(ax, sample, window[0], window[1], data_label)
          plt.show()

          fig, axs = plt.subplots(1, 2, figsize=(10, 5))
          qq_plot_innovation(axs[0], err, window, init, data_label)
          if partial: pacf_plot_innovation(axs[1], err, window, init, data_label, lags=lags)
          else: acf_plot_innovation(axs[1], err, window, init, data_label, lags=lags)
          fig.tight_layout()
          plt.show()
     
     return sample

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

################################
# Gradation Line Functionality #
################################


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=None, linewidth=3, alpha=1.0, add_colorbar=False, cbar_label=None):

     # Default colors equally spaced on [0,1]:
     if z is None:
          z = np.linspace(0.0, 1.0, len(x))

     # Special case if a single number:
     if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
          z = np.array([z])

     z = np.asarray(z)

     x_new, y_new, z_new = interpolation(x, y, z)

     if norm is None and len(z_new) > 1:
          norm = plt.Normalize(z_new[0], z_new[-1])
     if norm is None and len(z_new) == 1:
          norm = plt.Normalize(0, 1)

     segments = make_segments(x_new, y_new)
     lc = mcoll.LineCollection(segments, array=z_new, cmap=cmap, norm=norm,
                               linewidth=linewidth, alpha=alpha)

     #ax = plt.gca()
     ax.add_collection(lc)

     if add_colorbar:
          plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=cbar_label)

     return lc

def interpolation(x, y, t=None):

     path = mpath.Path(np.column_stack([x, y]))
     verts = path.interpolated(steps=3).vertices
     x_new, y_new = verts[:, 0], verts[:, 1]

     if t is None: return x_new, y_new

     path_t = mpath.Path(np.column_stack([t, t]))
     t_new = path_t.interpolated(steps=3).vertices[:, 0]
     return x_new, y_new, t_new

def make_segments(x, y):
     points = np.array([x, y]).T.reshape(-1, 1, 2)
     segments = np.concatenate([points[:-1], points[1:]], axis=1)
     return segments
