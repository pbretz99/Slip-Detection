# Libraries
import matplotlib.pyplot as plt
import numpy as np
from DLM import filter_sample

# Local Code
from DLM import filter_sample
from Utilities import load_data
from Paper1 import set_up_local_discount_filter, times_for_thresh
from Plotting import filter_plot, error_plot, sigma_plot, add_times_to_plot
from Times import get_times_from_vel, clean_times


# Plot Model results on sample
def quick_plot(axs, results, Data, data_label, init, final):

     filter_plot(axs[0], results.point_estimate(), Data, init, final, data_label)
     error_plot(axs[1], results.standardized_error(), init, final, data_label)
     sigma_plot(axs[2], results.var_point_estimate(), init, final, log_scale=True)

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

def run_Vel_example():

     Vel = load_data('xvelocity')

     init, final = 1, 1000
     ModelVel = set_up_local_discount_filter(Vel[init], omega=0.2582, df=0.8, alpha=2, beta=0.0001**2, J=2)
     results = filter_sample(ModelVel, Vel, init, final, forgetful=True, memory=50)
     slip_start, slip_end = get_times_from_sigma(np.log(results.var_point_estimate()), init, window_start=[-25, 10], window_end=[-10, 50], burn_in=100)

     fig, axs = plt.subplots(3, 1, figsize=(7, 7))
     quick_plot(axs, results, Vel, 'X Wall Velocity', init, final)
     for ax in axs:
          add_times_to_plot(ax, init, final, slip_start, c='red', ls='--')
          add_times_to_plot(ax, init, final, slip_end, c='green', ls='--')
     fig.tight_layout()
     plt.show()

def run_Vel_times():

     Vel = load_data('xvelocity')

     init, final = 1, len(Vel)
     ModelVel = set_up_local_discount_filter(Vel[init], omega=0.2582, df=0.8, alpha=2, beta=0.0001**2, J=2)

     num_slips = []
     mem_range = np.linspace(5, 200, num=100)
     for memory in mem_range:
          results = filter_sample(ModelVel, Vel, init, final, forgetful=True, memory=memory)
          slip_start, __ = get_times_from_sigma(np.log(results.var_point_estimate()), init, window_start=[-25, 10], window_end=[-10, 50], burn_in=100)
          num_slips.append(len(slip_start))
          print('Memory = %i, count = %i' %(memory, len(slip_start)))
     
     fig, ax = plt.subplots(figsize=(7, 7))
     ax.plot(mem_range, num_slips, c='steelblue')
     ax.set_ylim(bottom=0)
     ax.set_xlabel('$N_{mem}$')
     ax.set_ylabel('Slip Count')
     ax.set_title('Varying Memory')
     plt.show()


if __name__ == "__main__":

     #run_Vel_example()
     run_Vel_times()
