import sys
import json
import math
import os.path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot

dat = json.load(open(sys.argv[1]))
arena = madplot.Arena(dat)

ordered_trials = sorted(dat['coupled'].keys())

nplots = len(dat['coupled'])
ncols = int(math.sqrt(nplots)+0.5)
nrows = nplots/ncols

fig = plt.figure()
gs = gridspec.GridSpec(ncols, nrows)

pct_in_area_per_time = {} #bagname:(offset[],pct[])

#do the control (i.e. unpunished)
ldf,tdf,geom = madplot.load_bagfile(os.path.join(dat['_base'], dat['control']))
pct_in_area_per_time['unpunished'] = madplot.calculate_time_in_area(tdf, arena, geom)

for i,trial in enumerate(ordered_trials):
    bname = dat['coupled'][trial]

    p = os.path.join(dat['_base'], bname)
    ldf,tdf,geom = madplot.load_bagfile(p)

    pct_in_area_per_time[bname] = madplot.calculate_time_in_area(tdf, arena, geom)

    ax = fig.add_subplot(gs[i])
    madplot.plot_tracked_trajectory(ax, tdf,
            intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
            limits=arena.get_limits()
    )

    ax.set_title(os.path.basename(bname))

ax = plt.figure().add_subplot(111)
for trial in ordered_trials:
    bname = dat['coupled'][trial]
    offset,pct = pct_in_area_per_time[bname]
    ax.plot(offset, pct, linestyle='solid', marker='o', label=bname)

bname = 'unpunished'
offset,pct = pct_in_area_per_time[bname]
ax.plot(offset, pct, linestyle='solid', marker='+', label=bname)

ax.set_xlabel('time since trial start (s)')
ax.set_ylabel('time spent in target area (pct)')
ax.legend()

plt.show()
