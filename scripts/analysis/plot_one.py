import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

ax = plt.figure().add_subplot(111)
arena = madplot.Arena()

ldf,tdf,geom = madplot.load_bagfile(sys.argv[1])
madplot.plot_tracked_trajectory(ax, tdf,
            intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
            limits=arena.get_limits()
)

plt.show()
