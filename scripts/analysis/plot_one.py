import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

fig = plt.figure()

ax = fig.add_subplot(111)
arena = madplot.Arena()

ldf,tdf,hdf,geom = madplot.load_bagfile(sys.argv[1], arena)
madplot.plot_tracked_trajectory(ax, tdf,
            intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
            limits=arena.get_limits(),
            color='k',
)
ax.add_patch(arena.get_patch(color='k', alpha=0.1))

plt.show()
