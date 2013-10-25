import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

arena = madplot.Arena()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ldf,tdf,geom = madplot.load_bagfile(sys.argv[1], arena)

ax.add_patch(arena.get_patch(color='k', alpha=0.1))
madplot.plot_laser_trajectory(ax, ldf,
            limits=arena.get_limits()
)
ax.set_title("The effect of TTM tracking on laser position.\nValues required to hit the fly head.")
ax.legend()

fig.savefig('laser.png', bbox_layout='tight')

plt.show()
