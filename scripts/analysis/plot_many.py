import sys
import json
import math
import os.path
import cPickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot

def colors_hsv_circle(n, alpha=1.0):
    _hsv = np.dstack( (np.linspace(0,2/3.,n), [1]*n, [1]*n) )
    _rgb = matplotlib.colors.hsv_to_rgb(_hsv)
    return np.dstack((_rgb, [alpha]*n))[0]
    

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

def plot_data(path, dat, exps=('coupled','uncoupled')):
    arena = madplot.Arena(dat)

    exps = [e for e in exps if e in dat]
    exps_colors = [plt.cm.gnuplot(i) for i in np.linspace(0, 1.0, len(exps))]

    pct_in_area_per_time = {k:[] for k in exps}
    pct_in_area_per_time_lbls = {k:[] for k in exps}
    latency_to_first_contact = {k:[] for k in exps}

    for exp in exps:
        ordered_trials = dat[exp]

        if len(ordered_trials) <= 8:
            gs = gridspec.GridSpec(2, 4)
        elif len(ordered_trials) <= 12:
            gs = gridspec.GridSpec(3, 4)
        elif len(ordered_trials) <= 16:
            gs = gridspec.GridSpec(4, 4)
        else:
            raise Exception("yeah, this figure will be ugly")

        fig = plt.figure("%s Trajectories" % exp.title(), figsize=(16,8))

        for i,trial in enumerate(ordered_trials):
            label = trial.get('label',os.path.basename(trial['bag']))
            ldf, tdf, hdf, geom = trial['data']

            print exp.title(), "Trial %d" % i, label

            ax = fig.add_subplot(gs[i])
            madplot.plot_tracked_trajectory(ax, tdf,
                    limits=arena.get_limits()
            )

            patch = arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2)
            if patch is not None:
                ax.add_patch(patch)

            ax.set_title(label)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            pct_in_area_per_time_lbls[exp].append( label )
            pct_in_area_per_time[exp].append ( madplot.calculate_time_in_area(tdf, 300, interval=30) )

            latency_to_first_contact[exp].append(madplot.calculate_latency_to_stay(tdf, 20))

        #need more colors
        fig = plt.figure("%s Time in Area" % exp.title())
        ax = fig.add_subplot(1,1,1)

        colormap = plt.cm.gnuplot
        ax.set_color_cycle(colors_hsv_circle(len(ordered_trials)))

        for lbl,data in zip(pct_in_area_per_time_lbls[exp], pct_in_area_per_time[exp]):
            offset,pct = data
            #the last point is the total pct, use that later
            ax.plot(offset[:-1], pct[:-1], linestyle='solid', label=lbl)

        ax.legend()

    #check all trials have the same number
    trial_lens = set(map(len, (dat[e] for e in exps)))
    if len(trial_lens) != 1:
        raise Exception("experiments contain different numbers of trials")
    ntrials = trial_lens.pop()

    #plot time in area percent
    ind = np.arange(ntrials)  # the x locations for the groups
    width = 0.35              # the width of the bars

    fig = plt.figure("Time in Area")
    ax = fig.add_subplot(1,1,1)
    for i,exp in enumerate(exps):
        pcts = []
        for offset,pct in pct_in_area_per_time[exp]:
            assert offset[-1] == -1
            pcts.append(pct[-1])
        ax.bar(ind+(i*width), pcts, width, label=exp, color=exps_colors[i])

    ax.set_xlabel('Trial')
    ax.set_ylabel('Percentage of time spent in area')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( [str(i) for i in range(ntrials)] )
    ax.legend()

    #plot latency to first 20s in area
    fig = plt.figure("Latency to first 20s contact")
    ax = fig.add_subplot(1,1,1)
    for i,exp in enumerate(exps):
        stds = []
        means = []
        for tts in latency_to_first_contact[exp]:
            means.append(np.mean(tts))
            stds.append(np.std(tts))
        ax.bar(ind+(i*width), means, width, label=exp, color=exps_colors[i], yerr=stds)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Latency to first 20s contact')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( [str(i) for i in range(ntrials)] )
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


