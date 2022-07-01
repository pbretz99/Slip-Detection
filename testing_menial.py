import matplotlib.pyplot as plt
from Plotting import add_lettering

scale = 5
fig, axs = plt.subplots(3, 2, figsize=(scale, 1.5 * scale))
for i in range(3):
     axs[i,1].set_yticks([])
for i in range(2):
     for j in range(2):
          axs[i,j].set_xticks([])
num = 0
letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for i in range(3):
     for j in range(2):
          add_lettering(axs[i,j], letters[num], 0.8, 0.8)
          num += 1
plt.savefig('a_posteriori_diagnostics.png')
plt.show()
