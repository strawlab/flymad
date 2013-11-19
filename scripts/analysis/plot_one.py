import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

fig = plt.figure()

ax = fig.add_subplot(111)
arena = madplot.Arena()

ldf,tdf,hdf,geom = madplot.load_bagfile(sys.argv[1], arena)
madplot.plot_tracked_trajectory(ax, tdf, arena,
            color='k',
)
ax.add_patch(arena.get_patch(color='k', alpha=0.1))

patch = arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2)
if patch is not None:
    ax.add_patch(patch)

plt.show()
