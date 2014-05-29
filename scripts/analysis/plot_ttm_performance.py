import sys
import json
import math
import os.path
import collections
import cPickle
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import roslib; roslib.load_manifest('flymad')
import flymad.madplot as madplot

Err = collections.namedtuple('Err', 'err dt_thisfly dt_ttm')
Target = collections.namedtuple('Target', 'obj_id from_idx to_idx v ttm_err wf_err')

ttm_unit = 'mm'
ttm_conv = lambda x: np.array(x)*0.00416 #px -> mm

wide_unit = 'mm'
w_arena = madplot.Arena('mm')
wide_conv = w_arena.scale #lambda x: x*0.21077 #px -> mm

def prepare_data(arena, path):

    pool_df = madplot.load_bagfile_single_dataframe(path, arena, ffill=False, filter_short_pct=10.0)

    targets = pool_df['target_type'].dropna().unique()
    if len(targets) != 1:
        raise Exception("Only head or body may be targeted")
    if targets[0] == 1:
        target_name = "head"
    elif targets[0] == 2:
        target_name = "body"
    else:
        raise Exception("Only head or body may be targeted")

    ldf_all = pool_df[~pool_df['lobj_id'].isnull()]
    hdf_all = pool_df[~pool_df['h_framenumber'].isnull()]

    #the first fly targeted
    prev = pool_df['lobj_id'].dropna().head(1)
    prev_id = prev.values[0]
    prev_ix = prev.index[0]

    target_ranges = {
        (0,100):[],
    }

    for ix, row in pool_df.iterrows():
        lobj_id = row['lobj_id']
        if np.isnan(lobj_id):
            continue
        elif lobj_id != prev_id:
            df = pool_df[prev_ix:ix]

            #mean speed of fly over its whole turn
            v = df['v'].mean()

            #details of the TrackedObj message
            ldf = df[~df['lobj_id'].isnull()]

            #we want to know when we go to TTM mode
            gbs = ldf.groupby('mode')
            try:
                #this is the time we first switch to TTM
                to_ttm = gbs.groups[2.0][0]

                print "------------",v

                ttm_errs = []
                ttm_dts_thisfly = []
                ttm_dts = []
                wf_errs = []
                wf_dts_thisfly = []
                wf_dts = []

                #get the head detection data once we start ttm
                hdf = pool_df[to_ttm:ix]
                #extract only the head messages
                for i,(hix,hrow) in enumerate(hdf[~hdf['h_framenumber'].isnull()].iterrows()):

                    #skip when we miss the head
                    if hrow['head_x'] == 1e6:
                        continue

                    if target_name == "head":
                        ttm_err = math.sqrt( (hrow['head_x']-hrow['target_x'])**2 +\
                                             (hrow['head_y']-hrow['target_y'])**2)
                    elif target_name == "body":
                        ttm_err = math.sqrt( (hrow['body_x']-hrow['target_x'])**2 +\
                                             (hrow['body_y']-hrow['target_y'])**2)

                    #if the first value is < 10 it is likely actually associated
                    #with the last fly TTM, so ignore it....
                    if (i == 0) and (ttm_err < 100):
                        continue

                    #the time since we switched to this fly
                    thisfly_dt = (hix-prev_ix).total_seconds()
                    #the time since we switched to TTM
                    ttm_dt = (hix-to_ttm).total_seconds()
                    ttm_errs.append(ttm_err)
                    ttm_dts_thisfly.append(thisfly_dt)
                    ttm_dts.append(ttm_dt)

                    wdf = ldf[:hix].tail(1)
                    wf_err = math.sqrt(  (wdf['fly_x']-wdf['laser_x'])**2 +\
                                         (wdf['fly_y']-wdf['laser_y'])**2)

                    #the time since we switched to this fly
                    thisfly_dt = (wdf.index[0]-prev_ix).total_seconds()
                    #the time since we switched to TTM
                    ttm_dt = (wdf.index[0]-to_ttm).total_seconds()
                    wf_errs.append(wf_err)
                    wf_dts_thisfly.append(thisfly_dt)
                    wf_dts.append(ttm_dt)

                    print thisfly_dt, ttm_dt, ttm_err

                ttm_err = Err(ttm_errs, ttm_dts_thisfly, ttm_dts)
                wf_err = Err(wf_errs, wf_dts_thisfly, wf_dts)
                trg = Target(lobj_id, prev_ix, ix, v, ttm_err, wf_err)

            except KeyError:
                #never switched to TTM
                print "never switched to ttm"
                trg = None
                pass

            #except Exception:
            #    print "UNKNOWN ERROR"
            #    trg = None

            #classify the target into which speed range
            if trg is not None:
                for k in target_ranges:
                    vmin,vmax = k
                    if vmin < trg.v <= vmax:
                        target_ranges[k].append(trg)

            prev_id = lobj_id
            prev_ix = ix

    data = {'target_name':target_name,
            'pooldf':pool_df,
            'ldf':ldf_all,
            'hdf':hdf_all,
            'target_ranges':target_ranges}

    pkl_fname = "%s_%s.pkl" % (path, target_name)
    cPickle.dump(data, open(pkl_fname,'wb'), -1)

    return data

def load_data(arena, path):
    pkl_fname = glob.glob("%s_*.pkl" % path)[0]
    return cPickle.load(open(pkl_fname,'rb'))

def plot_data(arena, path, data):
    target_name = data['target_name']
    pool_df = data['pooldf']
    ldf = data['ldf']
    hdf = data['hdf']
    target_ranges = data['target_ranges']

    fig = plt.figure("Image Processing Time (%s)" %  target_name)
    ax = fig.add_subplot(1,1,1)
    ax.hist(hdf['h_processing_time'].values, bins=100)
    ax.set_xlabel('processing time (s)')
    ax.set_title("Distribution of time taken for %s detection" % target_name)
    fig.savefig('imgproc_%s.png' % target_name)

    fig = plt.figure("TTM Tracking", figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(arena.get_patch(color='k', alpha=0.1))
    madplot.plot_laser_trajectory(ax, ldf, arena)
    ax.set_title("The effect of TTM tracking on laser position.\n"\
                 "Values required to hit the fly %s" % target_name)
    ax.legend()
    fig.savefig('ttmeffect_%s.png' % target_name)

    all_v = []
    all_e = []

    #YAY finally plot
    for k in target_ranges:
        vmin,vmax = k

        fvels = "%s-%s px" % (vmin,vmax)

        figf = plt.figure("tFly %s/s" % fvels)
        axf = figf.add_subplot(1,1,1)
        axf_w = axf.twinx()
        figt = plt.figure("tTTM %s/s" % fvels)
        axt = figt.add_subplot(1,1,1)
        axt_w = axt.twinx()

        for trg in target_ranges[k]:
            #widefiled
            err = trg.wf_err
            axf_w.plot(err.dt_thisfly,
                       wide_conv(err.err),
                       'r', alpha=0.3)
            axt_w.plot(err.dt_ttm,
                       wide_conv(err.err),
                       'r', alpha=0.3)

            #ttm
            err = trg.ttm_err
            axf.plot(err.dt_thisfly,
                     ttm_conv(err.err),
                     'k', alpha=0.2)
            axt.plot(err.dt_ttm,
                     ttm_conv(err.err),
                     'k', alpha=0.2)

            all_v.append(trg.v)
            all_e.append(np.mean(ttm_conv(err.err)))

        for ax in (axf,axt):
            ax.set_xlim([0, 0.25])
            ax.set_xlabel('time (s)')
            ax.set_ylabel('TTM error (%s)' % ttm_unit)
            ax.set_ylim([0, 1.5])

        for ax in (axf_w,axt_w):
            ax.set_ylabel('deviation from WF COM (%s)' % wide_unit)
            ax.set_ylim([0, 4])

        for axttm,axwf in [(axt,axt_w),(axf,axf_w)]:
            axttm.set_zorder(axwf.get_zorder()+1) # put ax in front of ax2
            axttm.patch.set_visible(False)

        axf.set_title("Spacial accuracy of %s targeting (fly speed %s/s)\n"\
                      "(time since targeting this fly)" % (target_name,fvels))
        axt.set_title("Spacial accuracy of %s targeting (fly speed %s/s)\n"\
                      "(time since switching to TTM targeting)" % (target_name,fvels))

        figf.savefig(('tFly%s%s.png' % (fvels,target_name)).replace(' ',''))
        figt.savefig(('tTTM%s%s.png' % (fvels,target_name)).replace(' ',''))

    fig = plt.figure("Speed %s" % target_name)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(wide_conv(all_v),all_e)
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0, 2.5])
    ax.set_title("Spacial accuracy of %s targeting" % target_name)
    ax.set_xlabel('fly speed (%s/s)' % wide_unit)
    ax.set_ylabel('error (%s)' % ttm_unit)
    fig.savefig('flyv_errpx_%s.png' % target_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag file')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    #This arena is only for plotting
    arena = madplot.Arena(False)

    if args.only_plot:
        data = load_data(arena, path)
    else:
        data = prepare_data(arena, path)

    plot_data(arena, path, data)

    if args.show:
        plt.show()


