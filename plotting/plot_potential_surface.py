import matplotlib.pyplot as plt
import numpy as np
import os

potential_surface_file = "results/potential_surface/potential_surface.txt"
os.chdir(os.path.dirname(__file__)+"/..")

potential_surface = np.loadtxt(potential_surface_file)
potential_surface = potential_surface[potential_surface[:,0].argsort()]
fig, ax1 = plt.subplots()
#plt.rcParams["path.simplify"] = True
#plt.rcParams["agg.path.chunksize"] = 100
#plt.scatter(potential_surface[:,0], potential_surface[:,1])
color = 'tab:red'
ax1.set_xlabel('R')
ax1.set_ylabel('loss', color=color)
ax1.plot(potential_surface[100000:150000,0], potential_surface[100000:150000,1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('grad', color=color)  # we already handled the x-label with ax1
#ax2.set_ylim([-1e2, 1e2])
ax2.plot(potential_surface[100000:150000,0], potential_surface[100000:150000,2], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.close()

    