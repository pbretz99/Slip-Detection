
import numpy as np
import matplotlib.pyplot as plt

from Plotting import add_lettering

def add_line(ax, start, stop, pad=0, **kwargs):
     start, stop = np.array(start), np.array(stop)
     for x, y in [start, stop]:
          ax.scatter(x, y, c='steelblue')
     padded_start = stop + (1 - pad / 2) * (start - stop)
     padded_stop = start + (1 - pad / 2) * (stop - start)
     x = [point[0] for point in [padded_start, padded_stop]]
     y = [point[1] for point in [padded_start, padded_stop]]
     ax.plot(x, y, **kwargs)

def add_triangle(ax, points, pad=0, fill=None, **kwargs):
     for i in range(2):
          for j in range(i+1, 3):
               add_line(ax, points[i], points[j], pad=pad, **kwargs)
     if fill is not None:
          fill_triangle(ax, points, fill)

def add_tetrahedron(ax, back_points, front_points, pad=0, fill=None, **kwargs):
     triangle_1 = [back_points[0]] + front_points
     triangle_2 = [back_points[1]] + front_points
     for triangle in [triangle_1, triangle_2]:
          add_triangle(ax, triangle, pad=pad, fill=fill, **kwargs)
     add_line(ax, back_points[0], back_points[1], pad=pad, ls='--', **kwargs)


def fill_triangle(ax, points, fill):
     def key_func(point):
          return point[0]
     points = sorted(points, key=key_func)
     left, middle, right = points
     m = (right[1] - left[1]) / (right[0] - left[0])
     middle_on_line = [middle[0], m * (middle[0] - left[0]) + left[1]]
     x = [point[0] for point in points]
     y1 = [point[1] for point in points]
     y2 = [point[1] for point in [left, middle_on_line, right]]
     ax.fill_between(x, y1, y2, facecolor=fill['facecolor'], alpha=fill['alpha'])

def pad_axes(ax, pad=0.1):
     left, right = ax.get_xlim()
     x_d = right - left
     bottom, top = ax.get_ylim()
     y_d = top - bottom
     ax.set_xlim(left - pad * x_d, right + pad * x_d)
     ax.set_ylim(bottom - pad * y_d, top + pad * y_d)

def run_plot_simplices():

     scale = 4
     fig, axs = plt.subplots(1, 4, figsize=(5 * scale, scale))

     axs[0].scatter(0, 0)

     add_line(axs[1], [0, 0], [2, 1], c='steelblue')

     add_triangle(axs[2], [[0, 0], [1, 0.3], [0.5, 1]], fill={'facecolor': 'steelblue', 'alpha': 0.5}, c='steelblue')

     add_tetrahedron(axs[3], back_points=[[-0.1, 0.5], [1, 0.2]], front_points=[[0, 0], [0.3, 1]], fill={'facecolor': 'steelblue', 'alpha': 0.5}, c='steelblue')

     for ax in axs:
          pad_axes(ax)
          ax.set_xticks([])
          ax.set_yticks([])

     for ax, letter in zip(axs, ['(a)', '(b)', '(c)', '(d)']):
          add_lettering(ax, letter, 0.1, 0.85)

     plt.show()

def run_plot_filtration():

     scale = 4
     fig, axs = plt.subplots(1, 4, figsize=(5 * scale, scale))

     lb, rb, rt, lt = [0, 0], [1, 0], [1, 1], [0, 1]

     for x, y in [lb, rt]:
          axs[0].scatter(x, y, c='steelblue')
     
     axs[1].scatter(lb[0], lb[1], c='steelblue')
     add_triangle(axs[1], [rb, rt, lt], c='steelblue')

     axs[2].scatter(lb[0], lb[1], c='steelblue')
     add_triangle(axs[2], [rb, rt, lt], fill={'facecolor': 'steelblue', 'alpha': 0.5}, c='steelblue')

     add_line(axs[3], lb, rb, c='steelblue')
     add_triangle(axs[3], [rb, rt, lt], fill={'facecolor': 'steelblue', 'alpha': 0.5}, c='steelblue')

     pad_axes(axs[-1], pad=0.5)
     for ax in axs:
          ax.set_xlim(axs[-1].get_xlim())
          ax.set_ylim(axs[-1].get_ylim())
          ax.set_xticks([])
          ax.set_yticks([])

     for ax, letter in zip(axs, ['(a) $X^0$', '(b) $X^1$', '(c) $X^2$', '(d) $X^3$']):
          add_lettering(ax, letter, 0.1, 0.85)

     plt.show()


run_plot_filtration()
