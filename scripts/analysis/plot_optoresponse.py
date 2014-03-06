import re
import os.path
import collections
import glob
import pprint
import datetime
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import roslib; roslib.load_manifest('flymad')
import rosbag

import madplot
import generate_mw_ttm_movies
import flymad.trackingparams

def wrap_dtheta_plus_minus(dtheta, around=np.pi):
    """
    wraps an angle over the interval [-around,around)

    e.g. wrap_dtheta_plus_minus(dt, np.pi) returns an angle
    wrapped [-pi,pi)
    """

    if (dtheta>0):
        return math.fmod(dtheta+around, 2.0*around) - around
    else:
        return math.fmod(dtheta-around, 2.0*around) + around

def calc_dtheta(df):
    #naieve method for calculating dtheta assuming wraparound.

    dtheta = []
    dtheta_idx = []

    for i0,i1 in madplot.pairwise(df.iterrows()):
        t0idx,t0row = i0
        t1idx,t1row = i1
        t0 = t0row['theta']
        t1 = t1row['theta']

        dt = (t1-t0)
        #dtheta ranges from -pi/2.0 -> pi/2.0
        dt = wrap_dtheta_plus_minus(dt, around=np.pi/2.0)

        dtheta.append(dt)
        dtheta_idx.append(t0idx)

    return pd.Series(dtheta,index=dtheta_idx)

def prepare_data(arena, path, smoothstr, smooth):

    GENOTYPES = {"NINE":"*.bag",
                 "CSSHITS":"ctrl/CSSHITS/*.bag",
                 "NINGal4":"ctrl/NINGal4/*.bag"}

    CHUNKS = collections.OrderedDict()
    CHUNKS["1_loff+ve"] = (0,30)
    CHUNKS["2_loff-ve"] = (30,60)
    CHUNKS["3_lon-ve"] = (60,120)
    CHUNKS["4_lon+ve"] = (120,180)
    CHUNKS["5_loff+ve"] = (180,240)
    CHUNKS["6_loff-ve"] = (240,300)
    CHUNKS["7_loff+ve"] = (300,360)

    data = {gt:dict() for gt in GENOTYPES}
    for gt in GENOTYPES:
        pattern = path + GENOTYPES[gt]
        bags = glob.glob(pattern)
        for bag in bags:

            df = madplot.load_bagfile_single_dataframe(
                        bag, arena,
                        ffill=['laser_power','e_rotator_velocity_data'],
                        extra_topics={'/rotator/velocity':['data']},
                        smooth=smooth
            )

            #there might be multiple object_ids in the same bagfile for the same
            #fly, so calculate dtheta for all of them
            dts = []
            for obj_id, group in df.groupby('tobj_id'):
                if np.isnan(obj_id):
                    continue

                dts.append( calc_dtheta(group) )

            #join the dthetas vertically (axis=0) and insert as a column into
            #the full dataframe
            df['dtheta'] = pd.concat(dts, axis=0)

            #calculate mean_dtheta over each phase of the experiment
            t00 = df.index[0]
            for c in CHUNKS:
                s0,s1 = CHUNKS[c]

                t0 = datetime.timedelta(seconds=s0)
                t1 = datetime.timedelta(seconds=s1)

                mean_dtheta = df[t00+t0:t00+t1]["dtheta"].mean()

                try:
                    data[gt][c].append(mean_dtheta)
                except KeyError:
                    data[gt][c] = [mean_dtheta]

    return data

def plot_data(arena, path, smoothstr, data):

    COLORS = {"NINE":"r",
              "CSSHITS":"g",
              "NINGal4":"b"}

    pprint.pprint(data)

    fig = plt.figure('dtheta %s' % (smoothstr), figsize=(16,10))
    ax = fig.add_subplot(1,1,1)

    for gt in data:
        xdata = []
        ydata = []
        for xlabel in sorted(data[gt]):
            xloc = int(xlabel[0])
            for mean_dtheta in data[gt][xlabel]:
                xdata.append(xloc)
                ydata.append(mean_dtheta)

        print gt,xdata,ydata

        ax.plot(xdata,ydata,'o',color=COLORS[gt],markersize=5,label=gt)

    ax.legend()
    ax.set_xlim(0,9)

    ax.set_xticks([int(s[0]) for s in sorted(data["NINE"])])
    ax.set_xticklabels([s[2:] for s in sorted(data["NINE"])])

    figpath = os.path.join(path,'dtheta_%s.png' % (smoothstr))
    fig.savefig(figpath)
    print "wrote", figpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)

    args = parser.parse_args()
    path = args.path[0]

    assert os.path.isdir(path)

    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    arena = madplot.Arena('mm')

    data = prepare_data(arena, path, smoothstr, args.smooth)
    plot_data(arena, path, smoothstr, data)

    if args.show:
        plt.show()

