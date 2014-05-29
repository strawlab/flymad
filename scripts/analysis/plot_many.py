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
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import roslib; roslib.load_manifest('flymad')
import flymad.madplot as madplot

#FIXME: Smoothing and dt calculation calculation is not valid here, as multiple
#object ids are interleaved in the x_px and y_px arrays.... this will take a while
#to fix
#
#for learning it is fine to keep it all in pixels anyway
arena = madplot.Arena('mm')
smooth_trajectories = True
unit_ext = "%s%s" % (arena.unit,{True:'s',False:''}[smooth_trajectories])

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
                                    (bpath, arena),
                                    {'smooth':smooth_trajectories})

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
                bdat = jobs[bname].get()
                data.append( bdat )

#                dt = bdat[1]['t_dt'].values
#                print "\tdt",dt.mean(),"+/-",dt.std()
#                ax = plt.figure(bname).add_subplot(1,1,1)
#                ax.plot(dt)
#                ax = plt.figure(bname+"hist").add_subplot(1,1,1)
#                ax.hist(dt,bins=20)
#                plt.show(block=False)

            trials["data"] = madplot.merge_bagfiles(
                                        data,
                                        dat.get('_geom_must_intersect', True))

    with open(madplot.get_path(path, dat, "%s_%s.pkl" % (fname,unit_ext)), 'wb') as f:
        cPickle.dump(dat, f, -1)

    return dat

def load_data(path):
    dat = json.load(open(path))
    fname = os.path.splitext(os.path.basename(path))[0]
    with open(madplot.get_path(path, dat, "%s_%s.pkl" % (fname,unit_ext)), 'rb') as f:
        return cPickle.load(f)

def _label_rect_with_n(ax, rects, nens):
    for rect,n in zip(rects,nens):
        height = rect.get_height()
        #protect against no bar
        if not np.isnan(height):
            ax.text(rect.get_x()+rect.get_width()/2.,
                    1.05*height,
                    str(n), ha='center', va='bottom')

def _plot_bar_and_line(per_exp_data, exps, title, xlabel, ylabel, ind, width, ntrials, xticklabels, exps_colors, filename, plotdir, use_sem=True):
    figb = plt.figure(title)
    axb = figb.add_subplot(1,1,1)
    figl = plt.figure("%s L" % title)
    axl = figl.add_subplot(1,1,1)

    #save a json file
    data = {"treatment":[],'trial':[],'flyid':[],'value':[],'experiment':[]}
    for treatment in exps:
        for trialn,(vals,exp_ids) in enumerate(per_exp_data[treatment]):
            assert len(vals) == len(exp_ids)
            for flyid,(val,exp_id) in enumerate(zip(vals,exp_ids)):
                data['treatment'].append(treatment)
                data['trial'].append(trialn)
                data['flyid'].append(flyid)
                data['value'].append(val)
                data['experiment'].append(exp_id)
    dfname = os.path.join(plotdir,'%s.df' % filename)
    df = pd.DataFrame(data)
    df.save(dfname)
    print "\twrote", dfname

    for i,exp in enumerate(exps):
        means = []
        stds = []
        pooled_nens = []

        for j,(v,ids) in enumerate(per_exp_data[exp]):
            pooled_nens.append(len(v))
            means.append( np.mean(v) )

            std = np.std(v)
            if use_sem:
                stds.append(std/np.sqrt(len(v)))
            else:
                stds.append(std)

        rects = axb.bar(ind+(i*width), means, width, label=exp, color=exps_colors[i], yerr=stds)
        _label_rect_with_n(axb, rects, pooled_nens)

        axl.errorbar(ind, means, label=exp, color=exps_colors[i], yerr=stds)

    for ax in [axb, axl]:
        _ylabel = "%s +/- %s" % (ylabel, "SEM" if use_sem else "STD")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_ylabel)
        ax.legend()

    axl.set_xlim([0, ntrials-1])
    axl.set_xticks(range(ntrials))
    axl.set_xticklabels(xticklabels)

    axb.set_xticks(ind+(len(exps)/2.0 * width))
    axb.set_xticklabels(xticklabels)

    figb.savefig(os.path.join(plotdir,'%s.png' % filename))
    figl.savefig(os.path.join(plotdir,'%s_l.png' % filename))

def _get_plot_gs(ordered_trials):
    if len(ordered_trials) <= 8:
        gs = gridspec.GridSpec(2, 4)
    elif len(ordered_trials) <= 12:
        gs = gridspec.GridSpec(3, 4)
    elif len(ordered_trials) <= 16:
        gs = gridspec.GridSpec(4, 4)
    else:
        raise Exception("yeah, this figure will be ugly")
    return gs

def plot_data(path, dat, debug_plot):

    if os.path.isdir(path):
        plotdir = path
        conf = dat
    else:
        if path.endswith('.json'):
            conf = json.load(open(path))
        else:
            conf = dat
        plotdir = os.path.dirname(path)

    plot_exps = conf.get('_plot', ['coupled'])

    exps = [e for e in plot_exps if e in dat]
    exps_colors = [plt.cm.gnuplot(i) for i in np.linspace(0, 1.0, len(exps))]

    if '_ntrials' in conf:
        ntrials = conf['_ntrials']
    else:
        #check all trials have the same number
        trial_lens = set(map(len, (dat[e] for e in exps)))
        if len(trial_lens) != 1:
            raise Exception("experiments contain different numbers of trials")

        ntrials = trial_lens.pop()

    pct_in_area_total = {k:[] for k in exps}
    pct_in_area_per_time = {k:[] for k in exps}
    pct_in_area_per_time_lbls = {k:[] for k in exps}
    latency_to_first_contact = {k:[] for k in exps}
    velocity_outside_area = {k:[] for k in exps}
    velocity_inside_area = {k:[] for k in exps}
    path_length = {k:[] for k in exps}

    pct_in_area_per_time_bins = range(0,300,30)

    for exp in exps:
        ordered_trials = dat[exp][0:ntrials]

        gs = _get_plot_gs(ordered_trials)
        fig = plt.figure("%s Trajectories" % exp.title(), figsize=(16,8))

        gsh = _get_plot_gs(ordered_trials)
        figh = plt.figure("%s Trajectories (Hist)" % exp.title(), figsize=(16,8))

        for i,trial in enumerate(ordered_trials):
            label = trial['label']
            ldf, tdf, hdf, geom = trial['data']

            dbg_plot_title = " %s %s" % (exp,label)

            print exp.title(), label

            ax = fig.add_subplot(gs[i])
            madplot.plot_tracked_trajectory(ax, tdf, arena, debug_plot=debug_plot, title=dbg_plot_title)

            patch = arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.4, zorder=9)
            if patch is not None:
                ax.add_patch(patch)

            ax.add_patch(arena.get_patch(fill=False, color='k', zorder=10))

            ok = tdf.dropna(axis=0,how='any',subset=['x','y'])
            axh = figh.add_subplot(gs[i])
            axh.set_aspect('equal')
            xlim,ylim = arena.get_limits()
            axh.set_xlim(*xlim)
            axh.set_ylim(*ylim)

            #manually scale the lognorm
            lognorm = matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-3)
            #(counts, xedges, yedges, Image)
            _,_,_,_im = axh.hist2d(ok['x'],ok['y'],bins=30,range=(xlim,ylim), normed=True, norm=lognorm)

            l_f = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False)
            figh.colorbar(_im,ax=axh,format=l_f)

            for _ax in (ax,axh):
                _ax.set_title(label)
                _ax.xaxis.set_visible(False)
                _ax.yaxis.set_visible(False)

            pct_in_area_per_time_lbls[exp].append( label )

            pct_in_area_per_time[exp].append(
                        madplot.calculate_time_in_area(tdf, 300,
                                                       pct_in_area_per_time_bins))

            pct_in_area_total[exp].append(
                        madplot.calculate_pct_in_area_per_objid_only_vals(tdf))

            tts, vel_out, vel_in, path_l = madplot.calculate_latency_and_velocity_to_stay(
                                                tdf, holdtime=20,
                                                tout_reset_time=2, arena=arena, geom=geom,
                                                debug_plot=debug_plot, title=dbg_plot_title)

            latency_to_first_contact[exp].append(tts)
            velocity_outside_area[exp].append(vel_out)
            velocity_inside_area[exp].append(vel_in)
            path_length[exp].append(path_l)

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

    ind = np.arange(ntrials)  # the x locations for the groups
    width = 1.0 / len(exps)   # no gaps between
    width -= (0.2 * width)    # 20% gap
    ticklabels = [str(i) for i in range(ntrials)]

    #plot time in area percent
    _plot_bar_and_line(pct_in_area_total, exps,
                       'Time in Area', 'Trial', 'Percentage of time spent in area',
                        ind, width, ntrials,
                        ticklabels, exps_colors,
                        'timeinarea', plotdir)

    #plot latency to first 20s in area
    _plot_bar_and_line(latency_to_first_contact, exps,
                       'Latency to first 20s contact', 'Trial', 'Latency to first 20s contact',
                        ind, width, ntrials,
                        ticklabels, exps_colors,
                        'latency', plotdir)

    #velocity
    _plot_bar_and_line(velocity_outside_area, exps,
                       'Velocity Outside Area', 'Trial', 'Velocity Outside Area',
                        ind, width, ntrials,
                        ticklabels, exps_colors,
                        'velocity_out', plotdir)
    _plot_bar_and_line(velocity_inside_area, exps,
                       'Velocity Inside Area', 'Trial', 'Velocity Inside Area',
                        ind, width, ntrials,
                        ticklabels, exps_colors,
                        'velocity_in', plotdir)

    #path length
    _plot_bar_and_line(path_length, exps,
                       'Path Length', 'Trial', 'Path Length',
                        ind, width, ntrials,
                        ticklabels, exps_colors,
                        'path_length', plotdir)

def plot_data_trajectories(path, dat):

    if os.path.isdir(path):
        plotdir = path
        conf = dat
    else:
        if path.endswith('.json'):
            conf = json.load(open(path))
        else:
            conf = dat
        plotdir = os.path.dirname(path)

    plot_exps = conf.get('_plot', ['coupled'])

    exps = [e for e in plot_exps if e in dat]
    exps_colors = [plt.cm.gnuplot(i) for i in np.linspace(0, 1.0, len(exps))]

    if '_ntrials' in conf:
        ntrials = conf['_ntrials']
    else:
        #check all trials have the same number
        trial_lens = set(map(len, (dat[e] for e in exps)))
        if len(trial_lens) != 1:
            raise Exception("experiments contain different numbers of trials")

        ntrials = trial_lens.pop()

    plotdir = os.path.join(plotdir, 'trajectories')
    try:
        os.makedirs(plotdir)
    except OSError:
        pass

    for exp in exps:
        ordered_trials = dat[exp][0:ntrials]
        for i,trial in enumerate(ordered_trials):
            label = trial['label']
            ldf, tdf, hdf, geom = trial['data']

            dbg_plot_title = " %s %s" % (exp,label)

            good_oids = madplot.calculate_pct_in_area_per_objid(tdf)
            for _exp in good_oids:
                for _oid in good_oids[_exp]:
                    _fdf = tdf[(tdf['experiment'] == _exp) & (tdf['tobj_id'] == _oid)].resample('100L')
                    _fig = plt.figure(figsize=(2,2), frameon=False)
                    _ax = _fig.add_subplot(1,1,1)

                    _ax.set_aspect('equal')
                    _ax.xaxis.set_visible(False)
                    _ax.yaxis.set_visible(False)
                    patch = arena.get_intersect_patch(geom, fill=True, color='#1A9641', closed=True, alpha=0.4, zorder=9)
                    if patch is not None:
                        _ax.add_patch(patch)
                    _ax.add_patch(arena.get_patch(fill=False, color='k'))

                    xlim,ylim = arena.get_limits()
                    _ax.set_xlim(*xlim)
                    _ax.set_ylim(*ylim)
                    _ax.plot(_fdf['x'],_fdf['y'],color='#292724')

                    _fig.savefig(os.path.join(plotdir,
                                 "%s_t%s_%.0f_%s.svg" % (exp,i,good_oids[_exp][_oid],_oid)))
                    plt.close(_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1,
                        help='path to json file or directory of bag files from '\
                             'reiser experiments')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--debug-plot', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path)

    plot_data(path, data, args.debug_plot)
    #plot_data_trajectories(path, data)

    if args.show:
        plt.show()


