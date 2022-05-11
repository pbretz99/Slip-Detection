# Libraries
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# Local Code
from Utilities import load_data, print_tracker
from Times import get_times_from_error

class EpsNode:
     def __init__(self, level, t):
          '''
          :param level: the epsilon level of the detection
          :param t: the detection time
          '''
          self.level = level
          self.t = t
     
     def print(self):
          print(f'Detection at t = {self.t} at level eps = {self.level}')

     def point(self):
          return [self.t, self.level]

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
          for t in times_list[i]:
               nodes.append(EpsNode(levels[i], t))
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
               for t in bucket:
                    current_node = EpsNode(levels[i], t)
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
               if abs(node_1.t - node_2.t) <= R:
                    edges.append((node_1, node_2))
     return edges

def plot_component(component, levels, data, data_label):
     
     t, eps = nodes_to_points(component)
     init, final = np.min(t) - 100, np.max(t) + 100
     fig, axs = plt.subplots(2, 1)
     
     axs[0].scatter(t, eps, c='black', s=1)
     axs[0].set_xlim(init, final)
     axs[0].set_ylim(np.min(levels), np.max(levels))
     axs[0].set_title(f'{data_label} Detections by Threshold')
     axs[0].set_ylabel('Threshold $\epsilon$')
     axs[0].set_xlabel('t [frames]')
     
     plot_sample(axs[1], data, [init, final], c='gray')
     axs[1].set_title(data_label)
     axs[1].set_ylabel(data_label)
     axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
     axs[1].set_xlabel('t [frames]')
     
     fig.tight_layout()
     plt.show()

def nodes_to_points(nodes):
     t, eps = [], []
     for node in nodes:
          t.append(node.t)
          eps.append(node.level)
     return np.array(t), np.array(eps)

def plot_sample(ax, data, window, **kwargs):
     init, final = window
     ax.plot(range(init, final), data[init:final], **kwargs)

def get_ranges(nodes):
     t, eps = nodes_to_points(nodes)
     return [np.min(t), np.max(t)], [np.min(eps), np.max(eps)]

def print_components(components, min_lifespan=None):
     if min_lifespan is None:
          min_lifespan = -1
     text = ''
     for component in components:
          t_range, eps_range = get_ranges(component)
          if eps_range[1] - eps_range[0] > min_lifespan:
               text += f'\nTime Interval ({t_range[0]}, {t_range[1]}); Lifespan {round(eps_range[1]-eps_range[0], 2)}; Epsilon Interval ({round(eps_range[0], 2)}, {round(eps_range[1], 2)})'
     print(text)
     return text

def get_components(err, eps_range, R, window_size=25, verbose=True, bucketed=False):
     
     # Get detection times at epsilon levels
     if verbose: print('Extracting Detections:')
     times_array = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          times, __ = get_times_from_error(err, 1, eps_range[i], window_size=window_size)
          times = times.tolist()
          if bucketed:
               times = bucket_times(times, start=0)
          times_array.append(times)
     
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

def time_ranges(components, extend=0, single_points=False):
     intervals = []
     for component in components:
          if single_points:
               init, final = component, component
          else:
               init, final = get_ranges(component)[0]
          intervals.append([init-extend, final+extend])
     return intervals

def lifespans(components):
     eps_span = []
     for component in components:
          init, final = get_ranges(component)[1]
          eps_span.append([final-init])
     return eps_span

def overlapping(interval_1, interval_2):
     for first, second in [[interval_1, interval_2], [interval_2, interval_1]]:
          for time in first:
               if second[0] <= time <= second[1]:
                    return True
     return False

def split_by_overlap(components_1, components_2, extend=0, single_points=False):
     time_intervals_1, time_intervals_2 = time_ranges(components_1, extend=extend), time_ranges(components_2, extend=extend, single_points=single_points)
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

# Note: times must be ordered
def bucket_times(times, bucket_size=1000, empty_buckets=True, start=None):
     buckets = []
     current_bucket = []
     if start is None:
          init = (times[0] // bucket_size) * bucket_size
     else:
          init = start
     for t in times:
          if t < init + bucket_size:
               current_bucket.append(t)
          else:
               buckets.append(current_bucket)
               new_init = (t // bucket_size) * bucket_size
               if empty_buckets:
                    count = (new_init - init) // bucket_size - 1
                    for __ in range(count):
                         buckets.append([])
               current_bucket = [t]
               init = new_init
     buckets.append(current_bucket)
     return buckets

if __name__ == '__main__':

     Vel = load_data('xvelocity')
     W2 = load_data('w2_b0')
     init, final = [0, len(Vel)]
     vel_err = np.load('vel_err.npy')[init:final]
     w2_b0_err = np.load('w2_b0_err.npy')[init:final]
     eps_range = np.linspace(0, 2.5, 251)

     vel_components = get_components(vel_err, eps_range, 50, bucketed=True)
     
     vel_time_ranges = time_ranges(vel_components)
     vel_lifespans = lifespans(vel_components)
     lifespans = []
     max_vals = []
     for i in range(1, len(vel_time_ranges)):
          start = vel_time_ranges[i-1][0]
          end = vel_time_ranges[i][0]
          if end > start:
               M = np.max(Vel[start:end])
               if M > 0:
                    max_vals.append(M)
                    lifespans.append(vel_lifespans[i-1])
          else:
               print(f'Bad! Times 1 = ({vel_time_ranges[i-1][0]}, {vel_time_ranges[i-1][1]}), Times 2 = ({vel_time_ranges[i][0]}, {vel_time_ranges[i][1]})')
     
     fig, ax = plt.subplots()
     ax.scatter(lifespans, max_vals)
     ax.set_ylabel('Max. Velocity for Event')
     ax.set_xlabel('$\epsilon$-Lifespan')
     plt.show()

     fig, ax = plt.subplots()
     ax.scatter(lifespans, np.log(max_vals))
     ax.set_ylabel('Max. Velocity for Event (log scale)')
     ax.set_xlabel('$\epsilon$-Lifespan')
     plt.show()


     
     w2_components = get_components(w2_b0_err, eps_range, 50, bucketed=True)

     # Change unpacking to [overlapping_list, distinct_list]
     vel_overlap, w2_overlap, vel_distinct, w2_distinct = split_by_overlap(vel_components, w2_components, extend=25)
     print(f'\n{len(vel_components)} Velocity Detections, {len(vel_components)-len(vel_distinct)} Matched, and {len(vel_distinct)} Distinct')
     print('\nDistinct Vel Components:')
     print_components(vel_distinct)
     print(f'\n{len(w2_components)} W2B0 Detections, {len(w2_components)-len(w2_distinct)} Matched, and {len(w2_distinct)} Distinct')
     print('\nDistinct W2 Components:')
     print_components(w2_distinct)

     print('\nDistinct Vel Components (lifespan > 0.5):')
     print_components(vel_distinct, min_lifespan=0.5)
     print('\nDistinct W2 Components (lifespan > 0.5):')
     print_components(w2_distinct, min_lifespan=0.5)
