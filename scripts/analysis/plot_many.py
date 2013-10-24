import sys
import json
import math
import os.path
import cPickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot

def prepare_data(path):
    dat = json.load(open(path))
    arena = madplot.Arena(dat)

    for k in dat:
        if k.startswith("_"):
            continue
        for bag in dat[k]:
            bname = bag["bag"]
            bag["data"] = madplot.load_bagfile(
                                madplot.get_path(path, dat,bname),
                                arena
            )

    with open(madplot.get_path(path, dat,'data.pkl'), 'wb') as f:
        cPickle.dump(dat, f, -1)

    return dat

def load_data(path):
    dat = json.load(open(path))
    with open(madplot.get_path(path, dat,'data.pkl'), 'rb') as f:
        return cPickle.load(f)

def plot_data(path, dat):
    arena = madplot.Arena(dat)

    ordered_trials = dat['coupled']

    if len(ordered_trials) <= 8:
        gs = gridspec.GridSpec(2, 4)
    elif len(ordered_trials) <= 12:
        gs = gridspec.GridSpec(3, 4)
    elif len(ordered_trials) <= 16:
        gs = gridspec.GridSpec(4, 4)
    else:
        raise Exception("yeah, this figure will be ugly")

    fig = plt.figure("Trajectories", figsize=(16,8))

    pct_in_area_per_time = []
    pct_in_area_per_time_lbls = []

    for i,trial in enumerate(ordered_trials):
        label = trial.get('label',trial['bag'])
        ldf, tdf, geom = trial['data']

        ax = fig.add_subplot(gs[i])
        madplot.plot_tracked_trajectory(ax, tdf,
                intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2),
                limits=arena.get_limits()
        )
        ax.set_title(label)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        pct_in_area_per_time_lbls.append( label )
        pct_in_area_per_time.append ( madplot.calculate_time_in_area(tdf, 300, interval=30) )

    #need more colors
    fig = plt.figure("Time in Area")
    ax = fig.add_subplot(1,1,1)

    colormap = plt.cm.gnuplot
    ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, len(ordered_trials))])

    for lbl,data in zip(pct_in_area_per_time_lbls, pct_in_area_per_time):
        offset,pct = data
        ax.plot(offset, pct, linestyle='solid', label=lbl)

    ax.legend()

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


