import sys
import json
import math
import os.path
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import madplot

Target = collections.namedtuple('Target', 'obj_id from_idx to_idx v err dt_thisfly dt_ttm')

arena = madplot.Arena()
#pool_df = madplot.load_bagfile_single_dataframe(sys.argv[1], arena, ffill=False)
#pool_df.save('pool.df')

pool_df = pd.load('pool.df')

ldf = pool_df[~pool_df['lobj_id'].isnull()]

fig = plt.figure("TTM Tracking", figsize=(8,8))
ax = fig.add_subplot(1,1,1)

ax.add_patch(arena.get_patch(color='k', alpha=0.1))
madplot.plot_laser_trajectory(ax, ldf,
            limits=arena.get_limits()
)
ax.set_title("The effect of TTM tracking on laser position.\nValues required to hit the fly head.")
ax.legend()
fig.savefig('ttmeffect.png')

hdf = pool_df[~pool_df['h_framenumber'].isnull()]

fig = plt.figure("Image Processing Time")
ax = fig.add_subplot(1,1,1)
ax.hist(hdf['h_processing_time'].values, bins=100)
ax.set_xlabel('processing time (s)')
fig.savefig('imgproc.png')

#the first fly targeted
prev = pool_df['lobj_id'].dropna().head(1)
prev_id = prev.values[0]
prev_ix = prev.index[0]

target_ranges = {
    (0,4):[],
    (4,8):[],
    (8,30):[],
}

for ix, row in pool_df.iterrows():
    lobj_id = row['lobj_id']
    if np.isnan(lobj_id):
        continue
    elif lobj_id != prev_id:
        df = pool_df[prev_ix:ix]

        #mean velocity of fly over its whole turn
        v = df['v'].mean()

        #details of the TrackedObj message
        ldf = df[~df['lobj_id'].isnull()]

        #we want to know when we go to TTM mode
        gbs = ldf.groupby('mode')
        try:
            #this is the time we first switch to TTM
            to_ttm = gbs.groups[2.0][0]

            print "------------",v

            errs = []
            dts_thisfly = []
            dts_ttm = []

            #get the head detection data once we start ttm
            hdf = pool_df[to_ttm:ix]
            #extract only the head messages
            for i,(hix,hrow) in enumerate(hdf[~hdf['h_framenumber'].isnull()].iterrows()):

                #skip when we miss the head
                if hrow['head_x'] == 1e6:
                    continue

                err = math.sqrt( (hrow['head_x']-hrow['target_x'])**2 +\
                                 (hrow['head_y']-hrow['target_y'])**2)

                #if the first value is < 10 it is likely actually associated
                #with the last fly TTM, so ignore it....
                if (i == 0) and (err < 100):
                    continue

                #the time since we switched to this fly
                dt_thisfly = (hix-prev_ix).total_seconds()
                #the time since we switched to this fly
                dt_ttm = (hix-to_ttm).total_seconds()

                errs.append(err)
                dts_thisfly.append(dt_thisfly)
                dts_ttm.append(dt_ttm)

                print dt_thisfly, dt_ttm, err

            trg = Target(lobj_id, prev_ix, ix, v, errs, dts_thisfly, dts_ttm)

        except KeyError:
            #never switched to TTM
            print "never switched to ttm"
            trg = None
            pass

        except Exception:
            print "UNKNOWN ERROR"
            trg = None

        #classify the target into which velocity range
        if trg is not None:
            for k in target_ranges:
                vmin,vmax = k
                if vmin < trg.v <= vmax:
                    target_ranges[k].append(trg)

        prev_id = lobj_id
        prev_ix = ix

all_v = []
all_e = []

#YAY finally plot
for k in target_ranges:
    vmin,vmax = k

    fvels = "%s-%s px" % (vmin,vmax)

    figf = plt.figure("tFly %s/s" % fvels)
    axf = figf.add_subplot(1,1,1)
    figt = plt.figure("tTTM %s/s" % fvels)
    axt = figt.add_subplot(1,1,1)

    for trg in target_ranges[k]:
        axf.plot(trg.dt_thisfly, trg.err, 'k', alpha=0.2)
        axt.plot(trg.dt_ttm, trg.err, 'k', alpha=0.2)

        all_v.append(trg.v)
        all_e.append(np.mean(trg.err))

    for ax in (axf,axt):
        ax.set_xlim([0, 0.25])
        ax.set_xlabel('time (s)')
        ax.set_ylabel('error (px)')

    axf.set_title("Spacial accuracy of antenna targeting (fly velocity %s/s)\n"\
                  "(time since targeting this fly)" % fvels)
    axt.set_title("Spacial accuracy of antenna targeting (fly velocity %s/s)\n"\
                  "(time since switching to TTM targeting)" % fvels)

    figf.savefig('tFly%s.png' % fvels)
    figt.savefig('tTTM%s.png' % fvels)

fig = plt.figure("Velocity")
ax = fig.add_subplot(1,1,1)
ax.scatter(all_v, all_e)
ax.set_ylim([0, 400])
ax.set_xlim([0, 10])
ax.set_title("Spacial accuracy of antenna targeting")
ax.set_xlabel('fly velocity (px/s)')
ax.set_ylabel('error (px)')
fig.savefig('flyv_errpx.png')

plt.show()

