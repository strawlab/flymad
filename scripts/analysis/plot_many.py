import sys
import json
import math
import os.path
import cPickle
import argparse
import multiprocessing
import glob

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import madplot

def prepare_data(path):
    if os.path.isdir(path):
        path = path + "/"
        dat = {"coupled":[]}
        for trialn,b in enumerate(sorted(glob.glob(os.path.join(path,"*.bag")))):
            dat["coupled"].append({"bag":os.path.basename(b),
                                   "label":"trial %d" % trialn}
            )
        with open(os.path.join(path,"example.json"), "w") as f:
            json.dump(dat, f)
        fname = "example"
    else:
        dat = json.load(open(path))
        fname = os.path.splitext(os.path.basename(path))[0]

    arena = madplot.Arena(dat)

    jobs = {}
    pool = multiprocessing.Pool()

    for k in dat:
        if k.startswith("_"):
            continue
        for trialn,trials in enumerate(dat[k]):
            bags = trials["bag"]
            if not isinstance(bags,list):
                bags = [bags]
            else:
                if not "label" in trials:
                    print "WARNING: Trial missing label"

            for bname in bags:
                bpath = madplot.get_path(path, dat, bname)
                jobs[bname] = pool.apply_async(
                                    madplot.load_bagfile,
                                    (bpath, arena))

    pool.close()
    pool.join()

    for k in dat:
        if k.startswith("_"):
            continue
        for trialn,trials in enumerate(dat[k]):
            bags = trials["bag"]
            if not isinstance(bags,list):
                bags = [bags]

            data = []
            for bname in bags:
                print "merge", bname, "to trial", trialn
                data.append( jobs[bname].get() )
            trials["data"] = madplot.merge_bagfiles(
                                        data,
                                        dat.get('_geom_must_intersect', True))

    with open(madplot.get_path(path, dat, fname+".pkl"), 'wb') as f:
        cPickle.dump(dat, f, -1)

    return dat

def load_data(path):
    dat = json.load(open(path))
    fname = os.path.splitext(os.path.basename(path))[0]
    with open(madplot.get_path(path, dat, fname+".pkl"), 'rb') as f:
        return cPickle.load(f)

def plot_data(path, dat):
    arena = madplot.Arena(dat)

    if os.path.isdir(path):
        plotdir = path
    else:
        plotdir = os.path.dirname(path)

    plot_exps = data.get('_plot', ['coupled'])

    exps = [e for e in plot_exps if e in dat]
    exps_colors = [plt.cm.gnuplot(i) for i in np.linspace(0, 1.0, len(exps))]

    pct_in_area_total = {k:[] for k in exps}
    pct_in_area_per_time = {k:[] for k in exps}
    pct_in_area_per_time_lbls = {k:[] for k in exps}
    latency_to_first_contact = {k:[] for k in exps}

    pct_in_area_per_time_bins = range(0,300,30)

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
            label = trial['label']
            ldf, tdf, hdf, geom = trial['data']

            print exp.title(), label

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

            pct_in_area_per_time[exp].append(
                        madplot.calculate_time_in_area(tdf, 300,
                                                       pct_in_area_per_time_bins))

            pct_in_area_total[exp].append(
                        madplot.calculate_total_pct_in_area(tdf, 300))

            latency_to_first_contact[exp].append(
                        madplot.calculate_latency_to_stay(tdf, 20))

        fig.savefig(os.path.join(plotdir,'%s_trajectories.png' % exp))

        #need more colors
        fig = plt.figure("%s Time in Area" % exp.title())
        ax = fig.add_subplot(1,1,1)

        colormap = plt.cm.gnuplot
        ax.set_color_cycle(madplot.colors_hsv_circle(len(ordered_trials)))

        for lbl,pct in zip(pct_in_area_per_time_lbls[exp], pct_in_area_per_time[exp]):
            mean = pct.mean(axis=0)
            std = pct.mean(axis=0)
            ax.plot(pct_in_area_per_time_bins, mean, linestyle='solid', label=lbl)

        ax.legend()

        plt.savefig(os.path.join(plotdir,'%s_time.png' % exp))

    #check all trials have the same number
    trial_lens = set(map(len, (dat[e] for e in exps)))
    if len(trial_lens) != 1:
        raise Exception("experiments contain different numbers of trials")
    ntrials = trial_lens.pop()

    #plot time in area percent
    ind = np.arange(ntrials)  # the x locations for the groups
    width = 0.35              # the width of the bars

    figb = plt.figure("Time in Area")
    axb = figb.add_subplot(1,1,1)
    figl = plt.figure("Time in Area L")
    axl = figl.add_subplot(1,1,1)

    for i,exp in enumerate(exps):
        means = []
        stds = []
        for pct in pct_in_area_total[exp]:
            means.append( np.mean(pct) )
            stds.append( np.std(pct) )

        axb.bar(ind+(i*width), means, width, label=exp, color=exps_colors[i], yerr=stds)
        axl.errorbar(ind, means, label=exp, color=exps_colors[i], yerr=stds)

    axb.set_xticks(ind+width)
    for ax in [axb, axl]:
        ax.set_xlabel('Trial')
        ax.set_ylabel('Percentage of time spent in area')
        ax.set_xticklabels( [str(i) for i in range(ntrials)] )
        ax.legend()

    figb.savefig(os.path.join(plotdir,'timeinarea.png'))
    figl.savefig(os.path.join(plotdir,'timeinarea_l.png'))

    #plot latency to first 20s in area
    figb = plt.figure("Latency to first 20s contact")
    axb = figb.add_subplot(1,1,1)
    figl = plt.figure("Latency to first 20s contact L")
    axl = figl.add_subplot(1,1,1)

    for i,exp in enumerate(exps):
        means = []
        stds = []
        for tts in latency_to_first_contact[exp]:
            means.append(np.mean(tts))
            stds.append(np.std(tts))

        axb.bar(ind+(i*width), means, width, label=exp, color=exps_colors[i], yerr=stds)
        axl.errorbar(ind, means, label=exp, color=exps_colors[i], yerr=stds)

    axb.set_xticks(ind+width)
    for ax in [axb, axl]:
        ax.set_xlabel('Trial')
        ax.set_ylabel('Latency to first 20s contact')
        ax.set_xticklabels( [str(i) for i in range(ntrials)] )
        ax.legend()

    figb.savefig(os.path.join(plotdir,'latency.png'))
    figl.savefig(os.path.join(plotdir,'latency_l.png'))


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


