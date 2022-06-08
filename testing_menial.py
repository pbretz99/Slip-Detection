
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Plotting import add_subplot_axes

def generate_data(k=-0.1, size=1000):
     r = np.random.uniform(low=0.1, size=size) ** k - 1
     theta = np.random.uniform(low=0, high=np.pi/2, size=size)
     return r * np.cos(theta), r * np.sin(theta)

def run_test():

     x, y = generate_data()

     scale = 5
     fig = plt.figure(figsize=(scale, scale))
     ax = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=2)
     ax_left = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
     ax_bottom = plt.subplot2grid((3, 3), (2, 1), colspan=2)

     ax.scatter(x, y)
     sns.kdeplot(x=x, ax=ax_bottom)
     sns.kdeplot(y=y, ax=ax_left)
     
     ax_left.set_ylim(ax.get_ylim())
     ax_bottom.set_xlim(ax.get_xlim())

     ax_left.invert_xaxis()

     ax.set_xticks([])
     ax.set_yticks([])

     plt.show()

run_test()
