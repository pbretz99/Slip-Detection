
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
fig, axs = plt.subplots(2, 1)
sns.kdeplot(np.array(data), ax=axs[0], bw_adjust=0.5)
plt.show()