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

def plot_component(component, levels, data):
     t, eps = nodes_to_points(component)
     init, final = np.min(t) - 100, np.max(t) + 100
     fig, axs = plt.subplots(2, 1)
     axs[0].scatter(t, eps, c='black', s=1)
     axs[0].set_xlim(init, final)
     axs[0].set_ylim(np.min(levels), np.max(levels))
     plot_sample(axs[1], data, [init, final], c='gray')
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

Vel = load_data('xvelocity')
err = np.load('vel_err.npy')[0:10000]
eps_final = 2.5
eps_range = np.linspace(0, eps_final, round(eps_final * 100) + 1)
init, final = 0, len(err)
times_array = []
for eps in eps_range:
     print(eps)
     times, __ = get_times_from_error(err[init:final], 1, eps, window_size=100)
     times_array.append(times.tolist())

print('\n')

R_list = [5, 25, 50, 75, 100]
counts = []
for R in R_list:
     G = initialize_graph(times_array, eps_range, R, verbose=False)
     components = list(nx.connected_components(G))
     print(f'\nThere are {len(components)} connected components at R = {R}\n')
     #plot_component(components[0], eps_range, Vel)
     counts.append(len(components))

print(counts)
