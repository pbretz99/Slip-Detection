
# Libraries
import numpy as np

# Get times from labels function
def get_times_from_labels(labels):
     TimesLabels = []
     for i in range(1, len(labels)):
          if labels[i] == 2 and labels[i-1] == 0:
               TimesLabels.append(i)
     return np.array(TimesLabels)


def get_times_from_sigma(sigma, init, window_start, window_end, init_regime='Stick', burn_in=100):

     slip_start_times, slip_end_times = [], []
     W1, W2 = min(window_end[0], window_start[0]), max(window_end[1], window_start[1])
     regime = init_regime
     potential_slip_start_times, potential_slip_end_times = np.array([]).astype(int), np.array([]).astype(int)
     for i in range(burn_in-W1+1, len(sigma)-W2):
          
          slip_start_sample = sigma[(i+window_start[0]):(i+window_start[1])]
          slip_end_sample = sigma[(i+window_end[0]):(i+window_end[1])]

          if regime == 'Stick':

               # Collect potential slip starts
               if np.all(slip_start_sample >= sigma[i]):
                    potential_slip_start_times = np.concatenate((potential_slip_start_times, np.array([i])))
               
               # Stop at next potential slip end
               if np.all(slip_end_sample <= sigma[i]):
                    
                    # Record best slip start
                    if np.any(potential_slip_start_times):
                         slip_start_times.append(potential_slip_start_times[np.argmin(sigma[potential_slip_start_times])])
                    
                    # Reset regime and record potential slip end
                    regime = 'Slip'
                    potential_slip_end_times = np.array([i])
          
          if regime == 'Slip':

               # Collect potential slip ends
               if np.all(slip_end_sample <= sigma[i]):
                    potential_slip_end_times = np.concatenate((potential_slip_end_times, np.array([i])))
               
               # Stop at next potential slip start
               if np.all(slip_start_sample >= sigma[i]):

                    # Record best slip end
                    if np.any(potential_slip_end_times):
                         slip_end_times.append(potential_slip_end_times[np.argmax(sigma[potential_slip_end_times])])

                    # Reset regime and record potential slip start
                    regime = 'Stick'
                    potential_slip_start_times = np.array([i])

     # Final time
     if regime == 'Stick':
          slip_start_times.append(potential_slip_start_times[np.argmin(sigma[potential_slip_start_times])])
     if regime == 'Slip':
          slip_end_times.append(potential_slip_end_times[np.argmax(sigma[potential_slip_end_times])])

     return np.array(slip_start_times)+init, np.array(slip_end_times)+init

# Using estimated change, find raw times of changepoints
def get_times_from_vel(vel_est, init, threshold=0, burn_in=0):
     
     # Initialize
     mins, maxes = [], []
     if vel_est[burn_in] > threshold: region = 'Increasing'
     else: region = 'Decreasing'

     # Loop to find local mins and maxes
     for t in range(burn_in, len(vel_est)):

          if region == 'Decreasing' and vel_est[t] > threshold:
               region = 'Increasing'
               mins.append(t)

          elif region == 'Increasing' and vel_est[t] <= threshold:
               region = 'Decreasing'
               maxes.append(t)
     
     return np.array(mins) + init, np.array(maxes) + init

# Given times, keep changepoints below data_max, farther than dist_max from previous time
def clean_times(times, data, data_max, dist_max, look_back=5):
     new_times = []
     for t in times[times > look_back]:
          add_time = True
          ave_val = np.mean(data[(t-look_back):(t+1)])
          if ave_val >= data_max: add_time = False
          if new_times:
               if t - new_times[-1] < dist_max: add_time = False
          if add_time: new_times.append(t)
     return np.array(new_times)

# Function for matching times
def match_times(detected_times, times_for_matching, pad=0):
     matched_times = []
     unmatched_times = []
     t_prev = 0
     for t in times_for_matching:
          matched_num = 0
          for s in detected_times:
               if s >= t_prev+pad and s < t+pad:
                    matched_num += 1
          if matched_num > 0:
               matched_times = matched_times + [t] * matched_num
          else:
               unmatched_times.append(t)
          t_prev = t
     if len(matched_times) < len(detected_times):
          matched_times = matched_times + [times_for_matching[-1]] * (len(detected_times) - len(matched_times))
     return np.array(matched_times), np.array(unmatched_times)

# Examine this function closer
def get_indices_matched(diffs_to_labels, matched_W2_times_to_labels):
     ind_label_matched = []
     for i in range(len(diffs_to_labels)-1):
          match = True
          if diffs_to_labels[i] < -50: match = False
          #if diffs_to_labels[i] > 100: match = False
          if matched_W2_times_to_labels[i+1] == matched_W2_times_to_labels[i]: match = False
          ind_label_matched.append(match)
     ind_label_matched.append(False)
     ind_label_matched = np.array(ind_label_matched)
     ind_label_not_matched = np.array([not ind for ind in ind_label_matched])
     return ind_label_matched, ind_label_not_matched

# Print measures
def print_measures_from_times(TimesW2, TimesLabels, cut_off=100, pad=25):

     f_n_dict, f_p, advance_dict, counts_dict = get_all_measures_from_times(TimesW2, TimesLabels, cut_off=cut_off, pad=pad)

     print('Missed Rate: %2.3f percent, Total Missed: %i out of %i' %(f_n_dict['Missed'] * 100, counts_dict['Missed'], counts_dict['Labels']))
     print('False Positive Rate: %2.3f percent, Total Extra: %i out of %i' %(f_p * 100, counts_dict['W2'] - counts_dict['Matched'], counts_dict['W2']))

     # Advance notice stats
     print('%2.3f percent not in advance; %i detections not in advance' %(100 * f_n_dict['Just Missed'], counts_dict['Just Missed']))
     print('Average advance notice: %2.2f' %(advance_dict['Mean']))
     print('Median advance notice: %2.2f' %(advance_dict['Median']))
     print('Min advance notice: %2.2f' %(advance_dict['Min']))
     print('%2.3f percent more than %i frames before; %i times' %(f_n_dict['Past Cutoff'] * 100, cut_off, counts_dict['Past Cutoff']))
     print('Total False Negative Rate: %2.3f percent' %(100 * f_n_dict['Total']))

def get_all_measures_from_times(TimesW2, TimesLabels, cut_off=100, pad=25):

     # Match W2 times to slip beginnings
     matched_W2_times_to_labels, __ = match_times(TimesW2, TimesLabels, pad=pad)

     # Get times to next matched slip beginning
     diffs_to_labels = matched_W2_times_to_labels - TimesW2

     # Get indices of matched and unmatched W2 times
     ind_label_matched, __ = get_indices_matched(diffs_to_labels, matched_W2_times_to_labels)

     # Counts
     N_slips = len(TimesLabels)
     N_times = len(matched_W2_times_to_labels)
     N_matched = len(np.unique(matched_W2_times_to_labels))
     N_missed = N_slips - N_matched

     # False negatives and false positives
     f_n_missed = N_missed / N_slips
     f_p = 0
     if N_times > 0: f_p = 1 - N_matched / N_times

     # Advance notice stats
     vals = []
     f_n_cutoff, f_n_just_missed, min = 0, 0, 0
     ave, med = 0, 0
     N_just_missed, N_past_cutoff = 0, 0
     
     if N_times > 0:
          vals = diffs_to_labels[ind_label_matched]
          N_just_missed = len(vals[vals < 0])
          N_past_cutoff = len(vals[vals > cut_off])
          if len(vals) > 0:
               f_n_cutoff = N_past_cutoff / len(vals)
               f_n_just_missed = N_just_missed / len(vals)
               min = np.min(vals)
          good_vals = vals[(vals >= 0) & (vals <= cut_off)]
          if len(good_vals > 0):
               ave = np.mean(good_vals)
               med = np.median(good_vals)
     
     # Arrange dictionaries
     f_n_total = f_n_missed + f_n_cutoff + f_n_just_missed
     f_n_total = np.min((f_n_total, 1))
     f_n_dict = {'Missed': f_n_missed, 'Past Cutoff': f_n_cutoff, 'Just Missed': f_n_just_missed, 'Total': f_n_total}
     
     advance_dict = {'Mean': ave, 'Median': med, 'Min': min}
     counts_dict = {'Labels': N_slips, 'W2': N_times, 'Matched': N_matched, 'Missed': N_missed, 'Just Missed': N_just_missed, 'Past Cutoff': N_past_cutoff, 'Diffs': vals}
     
     return f_n_dict, f_p, advance_dict, counts_dict
