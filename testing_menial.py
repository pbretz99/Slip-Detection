
import numpy as np
from matplotlib import pyplot as plt

from Utilities import load_data
from Times import get_times_from_error, get_start_times


Vel = load_data('xvelocity')
vel_err = np.load('vel_err.npy')

err_detection, __ = get_times_from_error(vel_err, 1, 1, window_size=100)
err_start = err_detection
#err_start = get_start_times(err_detection, Vel, 0.002, window_size=5)

vel_detection, __ = get_times_from_error(Vel, 1, 0.002, window_size=100)
vel_start = vel_detection
#vel_start = get_start_times(vel_detection, Vel, 0.001, window_size=5)

print(len(err_start))
print(len(vel_start))

def matched_in_radius(times_1, times_2, R=50):
     match_1 = []
     match_2 = []
     for t in times_1:
          for s in times_2:
               if abs(t - s) <= R:
                    match_1.append(t)
                    match_2.append(s)
     unmatched = []
     for times, match in [[times_1, match_1], [times_2, match_2]]:
          unmatch = []
          for t in times:
               if t not in match:
                    unmatch.append(t)
          unmatched.append(unmatch)
     return [match_1, match_2], unmatched

match, unmatch = matched_in_radius(err_start, vel_start, R=50)
print(len(unmatch[0]))
print(len(unmatch[1]))

'''
slip_start, slip_end = get_times_from_error(vel_err, 1, 2, window_size=100)


def get_max_vels(Vel, slip_start, slip_end=None):
     max_vels = []
     if slip_end is not None:
          slip_end = slip_end[slip_end > slip_start[0]]
     for i in range(len(slip_start)-1):
          init = slip_start[i]
          if slip_end is None: final = slip_start[i+1]
          else: final = slip_end[i]
          max_vels.append(np.max(Vel[init:final]))
     return np.array(max_vels)

vals = get_max_vels(Vel, slip_start, slip_end)
print(sorted(vals)[0:25])
fig, ax = plt.subplots()
ax.hist(vals, edgecolor='black', alpha=0.7, bins=25)
plt.show()'''
