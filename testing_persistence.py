
# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Local code
from Times import get_times_from_error
from Utilities import load_data

# List of len(times_2) where each entry contains the subset of times_1 mapped that that time in times_2
def match_close_times(times_1, times_2, pad=10):
     matching = []
     prev_matches = []
     for t in times_2:
          current_matches = []
          for s in times_1:
               if abs(s - t) < pad and s not in prev_matches:
                    current_matches.append(s)
                    prev_matches.append(s)
          matching.append(current_matches)
     return matching

def update_pairs(persistence_pairs, prev_times, current_times, current_eps, pad=25):
     
     matches_list = match_close_times(prev_times, current_times, pad=pad)
     matched_times = []
     for i in range(len(matches_list)):
          
          # Get new pairs
          if len(matches_list[i]) == 0:
               persistence_pairs.append({'time': current_times[i], 'pair': [current_eps, np.inf]})
          
          else:
               # Update persisting element
               prev_t = persist_time(persistence_pairs, matches_list[i][0], current_times[i])
               matched_times.append(prev_t)

               # Kill merged elements
               for j in range(1, len(matches_list[i])):
                    prev_t = kill_time(persistence_pairs, matches_list[i][j], current_eps)
                    matched_times.append(prev_t)
     
     # Kill dead elements
     for t in prev_times:
          if t not in matched_times:
               kill_time(persistence_pairs, t, current_eps)

def persist_time(persistence_pairs, prev_t, current_t):
     for pair in persistence_pairs:
          if pair['time'] == prev_t:
               pair['time'] = current_t
               return prev_t

def kill_time(persistence_pairs, prev_t, current_eps):
     for pair in persistence_pairs:
          if pair['time'] == prev_t:
               pair['pair'][1] = current_eps
               return prev_t

def persistence(times_list, eps_range, pad=10, verbose=True):
     persistence_pairs = []
     if verbose: print(eps_range[0])
     for t in times_list[0]:
          persistence_pairs.append({'time': t, 'pair': [eps_range[0], np.inf]})
     for i in range(1, len(times_list)):
          if verbose: print(eps_range[i])
          update_pairs(persistence_pairs, times_list[i-1], times_list[i], eps_range[i], pad=pad)
     return persistence_pairs

def histogram_lifespan(ax, persistence_pairs, **kwargs):
     births, deaths = plotting_vals(persistence_pairs, eps_range)
     lifespans = np.array(deaths) - np.array(births)
     ax.hist(lifespans, **kwargs)
     ax.set_xlabel('Lifespan ($\epsilon$ length)')
     ax.set_ylabel('Count')

def plot_persistence(ax, persistence_pairs, eps_range, pad=0.15, **kwargs):
     births, deaths = plotting_vals(persistence_pairs, eps_range)
     ax.plot(eps_range, eps_range, c='gray', ls='--', alpha=0.5)
     ax.plot(eps_range, [eps_range[0]] * len(eps_range), c='gray', alpha=0.5)
     ax.plot([eps_range[0]] * len(eps_range), eps_range, c='gray', alpha=0.5)
     ax.scatter(births, deaths, **kwargs)
     ax.set_xlabel('Birth')
     ax.set_ylabel('Death')
     eps_lower, eps_upper = eps_range[0], eps_range[-1]
     dist = eps_upper - eps_lower
     eps_lower -= pad * dist
     eps_upper += pad * dist
     ax.set_xlim(left=eps_lower, right=eps_upper)
     ax.set_ylim(bottom=eps_lower, top=eps_upper)

def plotting_vals(persistence_pairs, eps_range):
     births, deaths = [], []
     for p in persistence_pairs:
          births.append(p['pair'][0])
          deaths.append(np.min([p['pair'][1], eps_range[-1]]))
     return births, deaths

def highlight_threshold(ax, threshold, eps_range, **kwargs):
     x = [eps_range[0], threshold]
     ax.fill_between(x, threshold, eps_range[-1], **kwargs)

def plot_samples(times, window=(-150, 150)):
     fig, axs = plt.subplots(2, len(times))
     for i in range(len(times)):
          init, final = times[i] + window[0], times[i] + window[1]
          ax = axs[0,i]
          ax.plot(range(init, final), Vel[init:final], c='gray')
          ax.set_ylabel('Velocity')
          ax.set_title('Wall Velocity')
          ax = axs[1,i]
          ax.plot(range(init, final), err[init:final], c='gray')
          ax.set_ylabel('Error')
          ax.set_title('Velocity Model Error')
     fig.tight_layout()
     plt.show()

Vel = load_data('xvelocity')
err = np.load('vel_err.npy')
plot_samples([250, 625, 1000])
#err = np.load('w2_b0_err.npy')
eps_final = 2.5
eps_range = np.linspace(0, eps_final, round(eps_final * 100) + 1)
init, final = 0, len(err)
times_array = []
for eps in eps_range:
     print(eps)
     times, __ = get_times_from_error(err[init:final], 1, eps, window_size=100)
     times_array.append(times.tolist())



'''
all_times = []
all_eps = []
for i in range(len(times_array)):
     all_eps += [eps_range[i]] * len(times_array[i])
     all_times += times_array[i]

fig, ax = plt.subplots()
ax.scatter(all_eps, all_times, s=1)
ax.set_ylim(init, final)
ax.set_xlabel('Threshold $\epsilon$')
ax.set_ylabel('Time')
ax.set_title('Detection Times by Threshold')
plt.show()'''

persistence_pairs = persistence(times_array, eps_range, pad=50, verbose=True)
print(f'\nRefinement: {eps_range[1]-eps_range[0]}\nCount of pairs: {len(persistence_pairs)}')

fig, ax = plt.subplots()
histogram_lifespan(ax, persistence_pairs, facecolor='lightblue', edgecolor='black', alpha=0.9, bins=20)
ax.set_title('Histogram of Lifespan (Death - Birth) of Velocity Detections')
plt.show()

for eps in [None, 0.4, 2]:
     fig, ax = plt.subplots(figsize=(5, 5))
     plot_persistence(ax, persistence_pairs, eps_range, alpha=0.2)
     if eps is not None:
          highlight_threshold(ax, eps, eps_range, edgecolor='darkgreen', facecolor='green', alpha=0.2)
     ax.set_title('Persistence of Velocity Detections under $\epsilon$ Variation')
     plt.show()

