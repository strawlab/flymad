import sys
import json
import math
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot


dat = json.load(open(sys.argv[1]))
arena = madplot.Arena(dat)

ordered_trials = sorted(dat['coupled'].keys())
fig = plt.figure(figsize=(16,8))

if len(ordered_trials) <= 8:
    gs = gridspec.GridSpec(2, 4)
elif len(ordered_trials) <= 12:
    gs = gridspec.GridSpec(3, 4)
elif len(ordered_trials) <= 16:
    gs = gridspec.GridSpec(4, 4)
else:
    raise Exception("yeah, this figure will be ugly")

pct_in_area_per_time = {} #bagname:(offset[],pct[])

def do_bagcalc(bname, label=None):
    if label is None:
        label = bname

    ldf,tdf,geom = madplot.load_bagfile(
                        os.path.join(dat['_base'], bname),
                        arena
    )

    pct_in_area_per_time[label] = madplot.calculate_time_in_area(tdf, 300, interval=30)

    madplot.calculate_time_to_area(tdf, 300)

    return ldf, tdf, geom

for i,trial in enumerate(ordered_trials):
    bname = dat['coupled'][trial]

    ldf,tdf,geom = do_bagcalc(bname)

    ax = fig.add_subplot(gs[i])
    madplot.plot_tracked_trajectory(ax, tdf,
            intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
            limits=arena.get_limits()
    )

    ax.set_title(os.path.basename(bname),
#            prop={'size':10}
    )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

##do the unpunished control
#ldf,tdf,geom = do_bagcalc(bname=dat['control'], label="unpunished")
#ax = fig.add_subplot(gs[i+1])
#madplot.plot_tracked_trajectory(ax, tdf,
#        intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
#        limits=arena.get_limits()
#)
#ax.set_title("unpunished",
##        prop={'size':10}
#)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

gs.tight_layout(fig, h_pad=1.8)
fig.savefig('traces.png', bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)

#need more colors
colormap = plt.cm.gnuplot
ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, len(ordered_trials))])

for trial in ordered_trials:
    bname = dat['coupled'][trial]
    offset,pct = pct_in_area_per_time[bname]
    ax.plot(offset, pct, linestyle='solid', label=bname)

#bname = 'unpunished'
#offset,pct = pct_in_area_per_time[bname]
#ax.plot(offset, pct, linestyle='solid', marker='+', label=bname)

ax.set_xlabel('time since trial start (s)')
ax.set_ylabel('time spent in target area (pct)')
ax.legend()

fig.savefig('times.png', bbox_layout='tight')

plt.show()
