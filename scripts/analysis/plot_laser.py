import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

arena = madplot.Arena()
ax = plt.figure().add_subplot(111)
ldf,tdf,geom = madplot.load_bagfile(sys.argv[1])
madplot.plot_laser_trajectory(ax, ldf,
            limits=arena.get_limits()
)

plt.show()
