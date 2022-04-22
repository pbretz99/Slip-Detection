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

def print_components(components):
     text = ''
     for component in components:
          t_range, eps_range = get_ranges(component)
          text += f'\nTime Interval ({t_range[0]}, {t_range[1]}); Lifespan {round(eps_range[1]-eps_range[0], 2)}; Epsilon Interval ({round(eps_range[0], 2)}, {round(eps_range[1], 2)})'
     print(text)
     return text

def get_components(err, eps_range, R, verbose=True):
     
     # Get detection times at epsilon levels
     if verbose: print('Extracting Detections:')
     times_array = []
     for i in range(len(eps_range)):
          if verbose: print_tracker(i, len(eps_range))
          times, __ = get_times_from_error(err, 1, eps_range[i], window_size=100)
          times_array.append(times.tolist())
     
     # Create graph
     G = initialize_graph(times_array, eps_range, R, verbose=verbose)
     
     # Get components in a list, sorted by smallest detection time
     def key_func(component):
          t_range, eps_range = get_ranges(component)
          return t_range[0]
     components = sorted(nx.connected_components(G), key=key_func)
     if verbose: print(f'\nThere are {len(components)} connected components at R = {R}\n')
     
     return components

def time_ranges(components, extend=0):
     intervals = []
     for component in components:
          init, final = get_ranges(component)[0]
          intervals.append([init-extend, final+extend])
     return intervals

def overlapping(interval_1, interval_2):
     for first, second in [[interval_1, interval_2], [interval_2, interval_1]]:
          for time in first:
               if second[0] <= time <= second[1]:
                    return True
     return False

def split_by_overlap(components_1, components_2, extend=0):
     time_intervals_1, time_intervals_2 = time_ranges(components_1, extend=extend), time_ranges(components_2, extend=extend)
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
     return overlapping_1, overlapping_2, not_overlapping_1, not_overlapping_2

if __name__ == '__main__':

     Vel = load_data('xvelocity')
     W2 = load_data('w2_b0')
     init, final = [0, 5000]
     vel_err = np.load('vel_err.npy')[init:final]
     w2_b0_err = np.load('w2_b0_err.npy')[init:final]
     eps_range = np.linspace(0, 2.5, 251)

     vel_components = get_components(vel_err, eps_range, 50)
     w2_components = get_components(w2_b0_err, eps_range, 50)
     vel_overlap, w2_overlap, vel_distinct, w2_distinct = split_by_overlap(vel_components, w2_components, extend=25)
     print(f'\n{len(vel_components)} Velocity Detections, {len(vel_components)-len(vel_distinct)} Matched, and {len(vel_distinct)} Distinct')
     print('\nDistinct Components:')
     print_components(vel_distinct)
     print(f'\n{len(w2_components)} W2B0 Detections, {len(w2_components)-len(w2_distinct)} Matched, and {len(w2_distinct)} Distinct')
     print('\nDistinct Components:')
     print_components(w2_distinct)
     
