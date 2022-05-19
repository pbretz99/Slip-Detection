# Libraries
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Local Code
from Utilities import load_data, print_tracker, unique, not_in, overlapping, extend
from Times import get_times_from_error

class EpsNode:
     def __init__(self, level, start, stop):
          '''
          :param level: the epsilon level of the detection
          :param start: the slip start time
          :param stop: the slip stop time
          '''
          self.level = level
          self.start = start
          self.stop = stop
     
     def print(self):
          print(f'Detected slip ({self.start}, {self.stop}) at level eps = {self.level}')

     def point(self):
          return [self.start, self.stop, self.level]
     
     def interval(self):
          return [self.start, self.stop]

def initialize_graph(times_list, levels, R, verbose=False):
     '''
     Note: make this smarter; add nodes in a quick manner when R increases
     :param times_list: list of lists of times over varying levels, length N
     :param levels: array of levels corresponding to entries in times_list, length N
     :param R: radius of intervals for intersection
     '''
     step = levels[1] - levels[0]
     tol = 0.01 * step
     nodes = []
     if verbose: print('Constructing nodes:')
     for i in range(len(times_list)):
          if verbose: print_tracker(i, len(times_list))
          for start, stop in times_list[i]:
               nodes.append(EpsNode(levels[i], start, stop))
     G = nx.Graph()
     G.add_nodes_from(nodes)
     count = 0
     if verbose: print('Constructing edges:')
     for N in G:
          if verbose: print_tracker(count, len(nodes))
          for M in G:
               add_edge = True
               if abs(M.level - N.level - step) > tol: add_edge = False
               if abs(M.t - N.t) > R: add_edge = False
               if add_edge:
                    G.add_edge(N, M)
          count += 1
     return G

def initialize_graph_bucketed(buckets_list, levels, R, verbose=False):
     nodes, node_bucket_list = construct_nodes_from_buckets(buckets_list, levels, verbose=verbose)
     edges = construct_edges_from_buckets(node_bucket_list, R, verbose=verbose)
     G = nx.Graph()
     G.add_nodes_from(nodes)
     G.add_edges_from(edges)
     return G

# Return list of all nodes and nodes in bucket structure
def construct_nodes_from_buckets(buckets_list, levels, verbose=False):
     nodes = []
     node_buckets_list = []
     if verbose: print('Constructing nodes:')
     for i in range(len(buckets_list)):
          if verbose: print_tracker(i, len(buckets_list))
          current_level_node_buckets = []
          for bucket in buckets_list[i]:
               current_node_bucket = []
               for start, stop in bucket:
                    current_node = EpsNode(levels[i], start, stop)
                    nodes.append(current_node)
                    current_node_bucket.append(current_node)
               current_level_node_buckets.append(current_node_bucket)
          node_buckets_list.append(current_level_node_buckets)
     return nodes, node_buckets_list

def construct_edges_from_buckets(node_buckets_list, R, verbose=False):
     edges = []
     if verbose: print('Constructing edges:')
     for level in range(1, len(node_buckets_list)):
          if verbose: print_tracker(level, len(node_buckets_list))
          prev_buckets, current_buckets = node_buckets_list[level-1], node_buckets_list[level]
          N1, N2 = len(prev_buckets), len(current_buckets)
          # First pass: prev <-> current
          for b in range(min(N1, N2)):
               edges = edges + edges_between_nodes(prev_buckets[b], current_buckets[b], R)
          # Second pass: prev-1 <-> current
          for b in range(1, min(N1+1,N2)):
               edges = edges + edges_between_nodes(prev_buckets[b-1], current_buckets[b], R)
          # Third pass: prev <-> current-1
          for b in range(1, min(N1,N2+1)):
               edges = edges + edges_between_nodes(prev_buckets[b], current_buckets[b-1], R)
     return edges

def edges_between_nodes(nodes_1, nodes_2, R):
     edges = []
     for node_1 in nodes_1:
          for node_2 in nodes_2:
               if overlapping(node_1.interval(), node_2.interval(), R=R):
                    edges.append((node_1, node_2))
     return edges

def nodes_to_points(nodes):
     start, stop, eps = [], [], []
     for node in nodes:
          start.append(node.start)
          stop.append(node.stop)
          eps.append(node.level)
     return np.array(start), np.array(stop), np.array(eps)

def plot_sample(ax, data, window, **kwargs):
     init, final = window
     ax.plot(range(init, final), data[init:final], **kwargs)

def get_ranges(nodes):
     start, stop, eps = nodes_to_points(nodes)
     return [np.min(start), np.max(stop)], [np.min(eps), np.max(eps)]

def print_components(components, min_lifespan=None):
     if min_lifespan is not None:
          components = remove_by_lifespan(components, min_lifespan)
     text = ''
     for component in components:
          t_range, eps_range = get_ranges(component)
          text += f'\nTime Interval ({t_range[0]}, {t_range[1]}); Lifespan {round(eps_range[1]-eps_range[0], 2)}; Epsilon Interval ({round(eps_range[0], 2)}, {round(eps_range[1], 2)})'
     print(text)
     return text

def remove_by_lifespan(components, min_lifespan):
     new_components = []
     for component in components:
          __, eps_range = get_ranges(component)
          if eps_range[1] - eps_range[0] >= min_lifespan:
               new_components.append(component)
     return new_components     

def get_components(err, eps_range, R, window_size=25, verbose=True, bucketed=True):
     
     # Get detection times at epsilon levels
     if verbose: print('Extracting Detections:')
     times_array = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, stop = get_times_from_error(err, 1, eps_range[i], window_size=window_size)
          pairs = pair_times(start, stop)
          if bucketed:
               pairs = bucket_times(pairs, bucket_start=0)
          times_array.append(pairs)
     
     # Create graph
     if bucketed:
          G = initialize_graph_bucketed(times_array, eps_range, R, verbose=verbose)
     else:
          G = initialize_graph(times_array, eps_range, R, verbose=verbose)
     
     # Get components in a list, sorted by smallest detection time
     def key_func(component):
          t_range, __ = get_ranges(component)
          return t_range[0]
     components = sorted(nx.connected_components(G), key=key_func)
     if verbose: print(f'\nThere are {len(components)} connected components at R = {R}\n')
     
     return components

def time_ranges(components, R=0, single_points=False):
     intervals = []
     for component in components:
          if single_points:
               init, final = component, component
          else:
               init, final = get_ranges(component)[0]
          intervals.append(extend([init, final], R))
     return intervals

def lifespans(components):
     eps_span = []
     for component in components:
          init, final = get_ranges(component)[1]
          eps_span.append([final-init])
     return eps_span

def eps_birth(components, descending=True):
     eps_birth = []
     for component in components:
          init, final = get_ranges(component)[1]
          if descending:
               eps_birth.append(final)
          else:
               eps_birth.append(init)
     return eps_birth

def pair_times(start, stop):
     if len(start) == 0 or len(stop) == 0:
          return np.array([])
     if stop[0] <= start[0]:
          stop = stop[1:]
     N = min(len(start), len(stop))
     start, stop = start[0:N], stop[0:N]
     return np.stack((start, stop), axis=-1)

def split_by_overlap(components_1, components_2, R=0, single_points=False):
     time_intervals_1, time_intervals_2 = time_ranges(components_1, R=R), time_ranges(components_2, R=R, single_points=single_points)
     overlapping_1, overlapping_2 = [], []
     for i in range(len(time_intervals_1)):
          for j in range(len(time_intervals_2)):
               if overlapping(time_intervals_1[i], time_intervals_2[j]):
                    overlapping_1.append(components_1[i])
                    overlapping_2.append(components_2[j])
     not_overlapping_1, not_overlapping_2 = [], []
     for component in components_1:
          if component not in overlapping_1:
               not_overlapping_1.append(component)
     for component in components_2:
          if component not in overlapping_2:
               not_overlapping_2.append(component)
     return [overlapping_1, overlapping_2], [not_overlapping_1, not_overlapping_2]

# Compare component to given slip times (note: possibly make functionality better)
def compare_to_slips(components, R=0):
     components_intervals = time_ranges(components, R=R)
     slip_intervals = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int).to_numpy().tolist()[:-1] # Note: detection is finicky with last slip
     overlapping_1, overlapping_2 = [], []
     advance_overlapping = []
     for j in range(len(slip_intervals)):
          for i in range(len(components_intervals)):
               if overlapping(components_intervals[i], slip_intervals[j]):
                    overlapping_1.append(components[i])
                    overlapping_2.append(slip_intervals[j])
                    if slip_intervals[j][0] >= components_intervals[i][0] and slip_intervals[j] not in advance_overlapping:
                         advance_overlapping.append(slip_intervals[j])
     not_overlapping_1, not_overlapping_2 = [], []
     for component in components:
          if component not in overlapping_1:
               not_overlapping_1.append(component)
     for interval in slip_intervals:
          if interval not in overlapping_2:
               not_overlapping_2.append(interval)
     return [overlapping_1, overlapping_2, advance_overlapping], [not_overlapping_1, not_overlapping_2]

# Note: times must be ordered
def bucket_times(times, bucket_size=1000, empty_buckets=True, bucket_start=None):
     '''
     :param times: list of slip intervals [start, stop]
     :param bucket_size: size of time ranges to construct buckets, default 1000
     :param empty_buckets: if no slip interval is in a bucket range, place an empty array [], default True
     :param bucket_start: if specified, initializes the first bucket at that time
     :return buckets: list of lists of time intervals; something like [[I0, I1], [I2], [], [I3, I4, I5], ..., [IN]]
     '''
     buckets = []
     current_bucket = []
     if bucket_start is None:
          bucket_start = (times[0,0] // bucket_size) * bucket_size
     for start, stop in times:
          if start < bucket_start + bucket_size:
               current_bucket.append([start, stop])
          else:
               buckets.append(current_bucket)
               new_bucket_start = (start // bucket_size) * bucket_size
               if empty_buckets:
                    count = (new_bucket_start - bucket_start) // bucket_size - 1
                    for __ in range(count):
                         buckets.append([])
               current_bucket = [[start, stop]]
               bucket_start = new_bucket_start
     buckets.append(current_bucket)
     return buckets

def plot_max_vel(ax, Vel, vel_components, plot_birth=True, plot_max_vel=True, log_scale=False, **kwargs):
     
     eps_measures, vel_measures = eps_vel_measures(Vel, vel_components)

     if plot_birth:
          eps_measure = eps_measures[1]
          x_label = '$\epsilon_{max}$'
     else:
          eps_measure = eps_measures[0]
          x_label = '$\epsilon$-Lifespan'
     
     if plot_max_vel:
          vel_measure = vel_measures[0]
          y_label = 'Max. Velocity in Slip'
     else:
          vel_measure = vel_measures[1]
          y_label = 'Slip Size'
     
     scatter_eps_vel(ax, eps_measure, vel_measure, x_label, y_label, log_scale=log_scale, **kwargs)

def eps_vel_measures(Vel, vel_components):
     vel_time_ranges = time_ranges(vel_components)
     vel_lifespans = lifespans(vel_components)
     vel_births = eps_birth(vel_components)
     eps_lifespans = []
     eps_births = []
     max_vals = []
     slip_sizes = []
     for interval, lifespan, birth in zip(vel_time_ranges, vel_lifespans, vel_births):
          start, end = interval
          if end > start:
               max_vals.append(np.max(Vel[start:end]))
               slip_sizes.append(np.sum(Vel[start:end]))
               eps_lifespans.append(lifespan)
               eps_births.append(birth)
          else:
               print(f'Bad! Component time range ({start}, {end}), Lifespan ({lifespan})')
     return [np.array(eps_lifespans), np.array(eps_births)], [np.array(max_vals), np.array(slip_sizes)]

def scatter_eps_vel(ax, x_data, y_data, x_label, y_label, log_scale=True, **kwargs):
     mask = np.array([True] * len(x_data))
     if log_scale:
          mask = (y_data > 0)
          y_label += ' (log scale)'
          y_data = np.log(y_data[mask])
     ax.scatter(x_data[mask], y_data, **kwargs)
     ax.set_ylabel(y_label)
     ax.set_xlabel(x_label)

def plot_sample_bars(Vel, err, init, final, eps_range, window_size=25, verbose=True, plot_vel=True, **kwargs):
     
     # Get detection times at epsilon levels
     if verbose: print('Extracting Detections:')
     times_array = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          start, stop = get_times_from_error(err[init:final], 1, eps_range[i], window_size=window_size)
          pairs = pair_times(start, stop)
          times_array.append(pairs)
     
     # Plot
     if plot_vel:
          fig, axs = plt.subplots(2, 1)
          ax = axs[0]
     else:
          fig, ax = plt.subplots()
     
     for i in range(len(eps_range)):
          for interval in times_array[i]:
               ax.plot(interval, [eps_range[i]] * 2, **kwargs)
     ax.set_xlabel('Time')
     ax.set_ylabel('Normalized Error Threshold $\epsilon$')
     ax.set_title('Slip Intervals by Threshold')

     if plot_vel:
          ax = axs[1]
          ax.plot(range(init, final), Vel[init:final], c='gray')
          axs[0].set_xlim(ax.get_xlim())
          ax.set_ylabel('Velocity')
          fig.tight_layout()

     plt.show()

def print_overlapping(vel_components, w2_components, R=0, min_lifespan=None, print_distinct=True):
     if min_lifespan is not None:
          vel_components = remove_by_lifespan(vel_components, min_lifespan)
          w2_components = remove_by_lifespan(w2_components, min_lifespan)
     __, distinct_comp = split_by_overlap(vel_components, w2_components, R=R)
     text = f'\nThere are {len(vel_components)} Velocity components and {len(w2_components)} W2B0 components'
     if min_lifespan is not None:
          text += f' with an epsilon lifespan larger than {min_lifespan}'
     text += '.'
     text += f'\nOf these, {len(distinct_comp[0])} Velocity components ({round(100 * len(distinct_comp[0]) / len(vel_components), 2)}%) and '
     text += f'{len(distinct_comp[1])} W2B0 components ({round(100 * len(distinct_comp[1]) / len(w2_components), 2)}%) were unmatched.'
     print(text)
     if print_distinct:
          for data_label, components in [['Velocity', distinct_comp[0]], ['W2B0', distinct_comp[1]]]:
               if len(components) > 0:
                    print(f'\nDistinct {data_label} Components:')
                    print_components(components)
     return distinct_comp[1]

def print_comparison(components, data_label, R=0, print_distinct=True):
     slip_intervals = pd.read_csv('slip_times.csv', names=['Start', 'End'], dtype=int).to_numpy()[:-1]
     overlapping_comp, distinct_comp = compare_to_slips(components, R=R)
     text = f'\nThere are {len(components)} {data_label} components and {len(slip_intervals)} Slips.'
     text += f'\nOf these, {len(distinct_comp[0])} {data_label} components ({round(100 * len(distinct_comp[0]) / len(components), 2)}%) and '
     text += f'{len(distinct_comp[1])} slips ({round(100 * len(distinct_comp[1]) / len(slip_intervals), 2)}%) were unmatched.'
     text += f'\nOf the matched slips, {len(overlapping_comp[2])} ({round(100 * len(overlapping_comp[2]) / (len(slip_intervals) - len(distinct_comp[1])), 2)}%) were detected in advance'
     print(text)
     if print_distinct:
          if len(distinct_comp[0]) > 0:
               print(f'\nDistinct {data_label} Components:')
               print_components(components)
          if len(distinct_comp[1]) > 0:
               print('Distinct Slip Intervals:')
               for start, stop in distinct_comp[1]:
                    print(f'({start}, {stop})')
     return overlapping_comp[0]

if __name__ == '__main__':

     Vel = load_data('xvelocity')
     W2 = load_data('w2_b0')
     init, final = [0, len(Vel)]
     vel_err = np.load('vel_err.npy')[init:final]
     w2_b0_err = np.load('w2_b0_err.npy')[init:final]
     eps_range = np.linspace(0, 5, 251)
     
     #plot_sample_bars(Vel, vel_err, 0, 2500, np.linspace(0.2, 5, 41), c='steelblue')
     #plot_sample_bars(W2, w2_b0_err, 0, 2500, np.linspace(0.4, 5, 41), c='steelblue')
     
     vel_components = get_components(vel_err, np.linspace(0.2, 5, 51), R=0, verbose=True)
     w2_components = get_components(w2_b0_err, np.linspace(0.4, 5, 51), R=0)

     slip_vel_components = unique(print_comparison(vel_components, 'Velocity', print_distinct=False))
     microslip_vel_components = not_in(vel_components, slip_vel_components)
     slip_w2_components = unique(print_comparison(w2_components, 'W2B0', print_distinct=False))


     nonslip_w2_components = print_overlapping(vel_components, w2_components, print_distinct=False)
     microslip_w2_components = not_in(w2_components, nonslip_w2_components + slip_w2_components)
     #print_overlapping(vel_components, w2_components, min_lifespan=0.01, print_distinct=False)
     #print_overlapping(vel_components, remove_by_lifespan(w2_components, 0.01), print_distinct=False)

     large_vel_components = remove_by_lifespan(vel_components, 1.5)
     print(f'\nVelocity breakdown: Large-scale slips count = {len(slip_vel_components)}, Small-scale slips count = {len(microslip_vel_components)}, Total = {len(vel_components)}')
     print(f'\nW2B0 breakdown: Large-scale slips count = {len(slip_w2_components)}, Small-scale slips count = {len(microslip_w2_components)}, Unmatched count = {len(nonslip_w2_components)}, Total = {len(w2_components)}')
     
     #for components, component_labels in zip([slip_vel_components, microslip_vel_components], ['Large-Scale', 'Small-Scale']):
     #     print(f'\nList of all Velocity {component_labels} detections:')
     #     print_components(components)
     
     #for components, component_labels in zip([slip_w2_components, microslip_w2_components, nonslip_w2_components], ['Large-Scale', 'Small-Scale', 'Unmatched']):
     #     print(f'\nList of all W2B0 {component_labels} detections:')
     #     print_components(components)
     

     for max_vel in [True, False]:
          fig, ax = plt.subplots()
          plot_max_vel(ax, Vel, vel_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5)
          plt.show()

          fig, ax = plt.subplots()
          plot_max_vel(ax, Vel, slip_vel_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5, c='steelblue', label='Matched')
          plot_max_vel(ax, Vel, microslip_vel_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5, c='orange', label='Unmatched')
          ax.legend()
          plt.show()

          fig, ax = plt.subplots()
          plot_max_vel(ax, Vel, slip_w2_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5, c='steelblue', label='Matched to Slips')
          plot_max_vel(ax, Vel, microslip_w2_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5, c='orange', label='Matched to Velocity Components')
          plot_max_vel(ax, Vel, nonslip_w2_components, plot_max_vel=max_vel, log_scale=True, alpha=0.5, c='red', label='Unmatched')
          ax.legend()
          plt.show()
