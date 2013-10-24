import json
import math
import os.path
import datetime
import collections
import cPickle
import argparse

import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches

import roslib; roslib.load_manifest('rosbag')
import rosbag

import madplot

def prepare_data(path):

    dat = json.load(open(path))
    arena = madplot.Arena(dat)

    fly_data = dat['data']
    bpath = dat.get('_base',os.path.abspath(os.path.dirname(path)))

    pooled_on = {k:[] for k in "axbh"}
    pooled_off = {k:[] for k in "axbh"}
    pooled_lon = {k:[] for k in "axbh"}

    for exp in fly_data:

        l_df, t_df, geom = madplot.load_bagfile(
                                    madplot.get_path(path,dat,exp["bag"]),
                                    arena
        )

        #find when the laser was on
        l_on = l_df[l_df['laser_power'] > 0]
        #time of first laser on
        l_on0 = l_df.index[0] + datetime.timedelta(seconds=30)

        #t_off = t_df.head(3000)
        #t_on = t_df.tail(3000)

        #fig = plt.figure()
        #ax = fig.gca()
        #t_df.plot(ax=ax)
        #fig.savefig("%s_%s.png" % (exp["bag"],exp["type"]), bbox_inches='tight')

        #the laser was off at the start and on at the end
        #tracking data when the laser was on 
        t_on = t_df[l_on0:]
        #tracking data when the laser was off
        t_off = t_df[:l_on0]

        pooled_on[exp["type"]].append(t_on)
        pooled_off[exp["type"]].append(t_off)
        pooled_lon[exp["type"]].append(l_on)

    cPickle.dump(pooled_on, open(os.path.join(bpath,'pooled_on.pkl'),'wb'), -1)
    cPickle.dump(pooled_off, open(os.path.join(bpath,'pooled_off.pkl'),'wb'), -1)
    cPickle.dump(pooled_lon, open(os.path.join(bpath,'pooled_lon.pkl'),'wb'), -1)

    return pooled_on, pooled_off, pooled_lon

def load_data(path):
    dat = json.load(open(path))
    bpath = dat.get('_base',os.path.dirname(path))

    return (
        cPickle.load(open(os.path.join(bpath,'pooled_on.pkl'),'rb')),
        cPickle.load(open(os.path.join(bpath,'pooled_off.pkl'),'rb')),
        cPickle.load(open(os.path.join(bpath,'pooled_lon.pkl'),'rb'))
    )

def plot_data(path, data):

    pooled_on, pooled_off, pooled_lon = data

    on_means = []
    off_means = []
    on_stds = []
    off_stds = []

    label_map = {'a':'Antenna','b':'Body','h':'Head','x':'Nothing'}
    labels = []

    #keep the same order
    for k in [t for t in 'ahbx' if len(pooled_on.get(t,[]))]:
        on = pd.concat(pooled_on[k])
        off = pd.concat(pooled_off[k])

        on_mean = on['v'].mean()
        off_mean = off['v'].mean()
        on_std = on['v'].std()
        off_std = off['v'].std()

        print "%s: on %.2f +/- %.2f off %.2f +/- %.2f" % (
                k, on_mean, on_std,
                off_mean, off_std
        )

        on_means.append(on_mean)
        off_means.append(off_mean)
        on_stds.append(on_std)
        off_stds.append(off_std)
        labels.append( label_map[k] )

    N = len(on_means)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35        # the width of the bars

    fig = plt.figure("Aversion")
    ax = fig.add_subplot(1,1,1)

    rects1 = ax.bar(ind, on_means, width, color='b', yerr=on_std, ecolor='k')
    rects2 = ax.bar(ind+width, off_means, width, color='r', yerr=off_std, ecolor='k')

    ax.set_ylim([0, 20])

    ax.set_ylabel('Speed (pixels/s) +/- STD')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( labels )
    ax.spines['bottom'].set_color('none') # don't draw bottom spine

    ax.legend( (rects1[0], rects2[0]), ('Laser On', 'Laser Off'), loc='upper right' )

    fig.savefig('aversion.svg', bbox_inches='tight')
    fig.savefig('aversion.png', bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to json files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path)

    plot_data(path, data)

    if args.show:
        plt.show()

