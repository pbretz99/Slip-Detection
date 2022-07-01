
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Plotting import add_lettering

class Normal:
     def __init__(self, mean, scale):
          self.mean = mean
          self.scale = scale
     
     @property
     def var(self):
          return self.scale ** 2
     
     def max(self):
          return 1 / np.sqrt(2 * np.pi * self.var)
     
     def print(self):
          print(f'A normal distribution with mean = {round(self.mean, 2)} and std. = {round(self.scale, 2)} (var. = {round(self.var, 2)})')

     def plot(self, ax, lower, upper, num=101, **kwargs):
          x = np.linspace(lower, upper, num=num)
          ax.plot(x, norm.pdf(x, self.mean, self.scale), **kwargs)
     
     def fill(self, ax, lower, upper, num=101, **kwargs):
          x = np.linspace(lower, upper, num=num)
          ax.fill_between(x, norm.pdf(x, self.mean, self.scale), **kwargs)

     def update_with_likelihood(self, observation, obs_var, print_K=False):
          K = self.var / (self.var + obs_var)
          if print_K:
               print(f'The weight K = {round(K, 3)}')
          var_new = (1 - K) * self.var
          mean_new = self.mean + K * (observation - self.mean)
          return Normal(mean_new, np.sqrt(var_new))

def run_conjugate_example(prior_mean=65, prior_scale=5, observation=72.3, obs_scale=2):
     
     # Run the conjugate analysis
     Prior = Normal(prior_mean, prior_scale)
     Likelihood = Normal(observation, obs_scale) # This is one of the few cases where the likelihood function happens to be a distribution
     Posterior = Prior.update_with_likelihood(observation, obs_scale ** 2, print_K=True)
     Distributions = [Prior, Likelihood, Posterior]
     print('The posterior distribution:')
     Posterior.print()

     # Plot
     scale = 5
     fig, axs = plt.subplots(3, 1, figsize=(scale, scale))

     max_vals = [Distribution.max() for Distribution in Distributions]
     for ax, Distribution, color, letter in zip(axs, Distributions, ['steelblue', 'orange', 'green'], ['(a)', '(b)', '(c)']):
          Distribution.fill(ax, lower=50, upper=80, facecolor=color, edgecolor=color, alpha=0.7)
          ax.axvline(Distribution.mean, ls='--', c='black')
          ax.set_ylim(top=1.1 * max(max_vals))
          add_lettering(ax, letter, 0.05, 0.75)
          ax.set_ylabel('Density')

     for ax in axs[:-1]:
          ax.set_xticks([])
     
     axs[-1].set_xlabel('$^\circ$F')

     plt.show()

if __name__ == '__main__':
     run_conjugate_example()
