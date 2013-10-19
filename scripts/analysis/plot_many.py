import sys
import json
import math
import os.path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot

dat = json.load(open(sys.argv[1]))
arena = madplot.Arena(dat)

nplots = 1 + len(dat['coupled'])
ncols = int(math.sqrt(nplots)+0.5)
nrows = nplots/ncols

fig = plt.figure()
gs = gridspec.GridSpec(nrows, ncols)

for i,trial in enumerate(sorted(dat['coupled'].keys())):
    p = os.path.join(dat['_base'],dat['coupled'][trial])
    ax = fig.add_subplot(gs[i])

    ldf,tdf,geom = madplot.load_bagfile(p)

    madplot.plot_tracked_trajectory(ax, tdf,
            intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2)
    )

plt.show()
