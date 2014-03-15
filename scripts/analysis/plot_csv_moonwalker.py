import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import glob
import pickle
import math
import itertools
import re
import operator

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt
import matplotlib.image as mimg

import scipy.signal
from scipy.stats import ttest_ind

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import madplot

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0", "0.12.0")

HEAD    = +100
THORAX  = -100
OFF     = 0

EXPERIMENT_DURATION = 130.0

YLIM = [-10, 30]
YTICKS = [-20, 0, 20, 40]

XLIM = [-1, 9]
XTICKS = [0, 4, 8]

def prepare_data(path, arena, smooth, medfilt, only_laser, gts):

    LASER_THORAX_MAP = {True:THORAX,False:HEAD}

    #PROCESS SCORE FILES:
    pooldf = pd.DataFrame()
    for csvfile in sorted(glob.glob(path + "/*.csv")):

        #don't waste time smoothing files not in out genotype list
        _,_,_,_genotype,_laser,_ = flymad_analysis.extract_metadata_from_filename(csvfile)
        if _laser != only_laser:
            print "\tskipping laser", _laser, "!=", only_laser
            continue

        if _genotype not in gts:
            print "\tskipping genotype", _genotype, "!=", gts
            continue

        csvfilefn = os.path.basename(csvfile)
        cache_args = csvfilefn, arena, smoothstr
        cache_fname = csvfile+'.madplot-cache'

        results = madplot.load_bagfile_cache(cache_args, cache_fname)
        if results is None:
            results = flymad_analysis.load_and_smooth_csv(csvfile, arena, smooth)
            if results is not None:
                #update the cache
                madplot.save_bagfile_cache(results, cache_args, cache_fname)
            else:
                print "skipping", csvfile
                continue

        df,dt,experimentID,date,time,genotype,laser,repID = results

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < EXPERIMENT_DURATION:
            print "\tmissing data", csvfilefn
            continue

        print "\t%ss experiment" % duration

        #we use zx to rotate by pi
        df['zx'][df['zx'] > 0] = math.pi

        #ROTATE by pi if orientation is east
        df['orientation'] = df['theta'] + df['zx']

        #ROTATE by pi if orientation is north/south (plusminus 0.25pi) and hemisphere does not match scoring:
        smask = df[df['as'] == 1]
        smask = smask[smask['orientation'] < 0.75*(math.pi)]
        smask = smask[smask['orientation'] > 0.25*(math.pi)]
        amask = df[df['as'] == 0]
        amask1 = amask[amask['orientation'] > -0.5*(math.pi)]
        amask1 = amask1[amask1['orientation'] < -0.25*(math.pi)]
        amask2 = amask[amask['orientation'] > 1.25*(math.pi)]
        amask2 = amask2[amask2['orientation'] < 1.5*(math.pi)]
        df['as'] = 0
        df['as'][smask.index] = math.pi
        df['as'][amask1.index] = math.pi
        df['as'][amask2.index] = math.pi
        df['orientation'] = df['orientation'] - df['as']
        df['orientation'] = df['orientation'].astype(float)

        df['orientation'][np.isfinite(df['orientation'])] = np.unwrap(df['orientation'][np.isfinite(df['orientation'])]) 
        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 300] = np.nan

        #CALCULATE FORWARD VELOCITY
        df['Vtheta'] = np.arctan2(df['vy'], df['vx'])
        df['Vfwd'] = (np.cos(df['orientation'] - df['Vtheta'])) * df['v']
        df['Afwd'] = np.gradient(df['Vfwd'].values) / dt
        df['dorientation'] = np.gradient(df['orientation'].values) / dt

        try:
            df = flymad_analysis.align_t_by_laser_on(
                    df, min_experiment_duration=EXPERIMENT_DURATION,
                    align_first_only=False,
                    t_range=(-1,9),
                    min_num_ranges=5)
        except flymad_analysis.AlignError, err:
            print "\talign error %s (%s)" % (csvfilefn, err)
            continue

        #median filter
        if medfilt:
            df['Vfwd'] = scipy.signal.medfilt(df['Vfwd'].values, medfilt)

        df['obj_id'] = flymad_analysis.create_object_id(date,time)
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        pooldf = pd.concat([pooldf, df]) 

    data = {}
    for gt in gts:
        gtdf = pooldf[pooldf['Genotype'] == gt]

        lgs = gtdf['lasergroup'].unique()
        if len(lgs) != 1:
            raise Exception("only one lasergroup handled for gt %s: not %s" % (
                             gt, lgs))

        grouped = gtdf.groupby(['t'], as_index=False)

        data[gt] = dict(mean=grouped.mean().astype(float),
                        std=grouped.std().astype(float),
                        n=grouped.count().astype(float),
                        first=grouped.first(),
                        df=gtdf)

    return data

def plot_cross_activation_only(path, data, arena, note):

    PLOTS = [('50660chrim',{'activation':'350ru','cross_activation':'350iru'}),
             ('50660trp',{'activation':'434iru','cross_activation':'350ru'}),
             ('50660trp',{'activation':'350iru','cross_activation':'350ru'}),
    ]

    for gt,lasers in PLOTS:
        figname = 'vs'.join(lasers.values())

        alaser = lasers['activation']
        adf = data[gt][alaser][gt]
        claser = lasers['cross_activation']
        cdf = data[gt][claser][gt]
        datasets = {
            'activation':dict(xaxis=adf['mean']['t'].values,
                              value=adf['mean']['Vfwd'].values,
                              std=adf['std']['Vfwd'].values,
                              n=adf['n']['Vfwd'].values,
                              label='Activation (%s)' % flymad_analysis.laser_desc(alaser),
                              order=0,
                              color=flymad_plot.RED,
                              N=len(adf['df']['obj_id'].unique()),
                              df=adf['df']),
            'cross_activation':dict(
                              xaxis=cdf['mean']['t'].values,
                              value=cdf['mean']['Vfwd'].values,
                              std=cdf['std']['Vfwd'].values,
                              n=cdf['n']['Vfwd'].values,
                              label='Cross Activation (%s)' % flymad_analysis.laser_desc(claser),
                              order=1,
                              color=flymad_plot.BLACK,
                              N=len(cdf['df']['obj_id'].unique()),
                              df=cdf['df']),
        }

        figure_title = "Moonwalker Thorax Crosstalk %s (%s)" % (gt,figname)

        fig = plt.figure(figure_title, figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        l,axs, figs = flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=dict(xaxis=adf['first']['t'].values,
                                       where=adf['first']['laser_state'].values>0),
                    sem=True,
                    downsample=5,
                    note="%s\n%s\n" % (gt,note),
                    individual={k:{'groupby':('obj_id','trial'),'xaxis':'t','yaxis':'Vfwd'} for k in datasets},
                    individual_title=figure_title + ' Individual Traces',
                    **datasets
        )

        ax.axhline(color='k', linestyle='--',alpha=0.8)
        ax.set_ylim(YLIM)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fwd Velocity (%s/s)' % arena.unit)
        ax.set_xlim(XLIM)

        flymad_plot.retick_relabel_axis(ax, XTICKS, YTICKS)

        fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s_%s.png" % (gt, figname)), bbox_inches='tight')
        fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s_%s.svg" % (gt, figname)), bbox_inches='tight')

        for efigname, efig in figs.iteritems():
            efig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s_%s_individual_%s.png" % (gt, figname, efigname)), bbox_inches='tight')


def plot_all_data(path, data, arena, note):

    for gt in data:
        datasets = {}
        color_cycle = itertools.cycle(flymad_plot.EXP_COLORS)

        laser_powers_sorted = sorted(data[gt], cmp=flymad_analysis.cmp_laser, reverse=True)

        for order,laser in enumerate(laser_powers_sorted):

            #also sort to make cross-activation first
            cross_activation = (re.match('[0-9]+iru$',laser) and gt.endswith('chrim')) or \
                               (re.match('[0-9]+ru$',laser) and gt.endswith('trp'))

            gtdf = data[gt][laser][gt]

            laser_desc = flymad_analysis.laser_desc(laser)
            datasets[laser] = dict(xaxis=gtdf['mean']['t'].values,
                                   value=gtdf['mean']['Vfwd'].values,
                                   std=gtdf['std']['Vfwd'].values,
                                   n=gtdf['n']['Vfwd'].values,
                                   label=laser_desc,
                                   order=50 if cross_activation else order,
                                   color=color_cycle.next(),
                                   N=len(gtdf['df']['obj_id'].unique()),
                                   df=gtdf['df'],
            )

        if 1:
            if gt == "50660chrim":
                cgt = "50660"
                gtdf = data[cgt]["350iru"][cgt]
                datasets['control'] = dict(xaxis=gtdf['mean']['t'].values,
                                   value=gtdf['mean']['Vfwd'].values,
                                   std=gtdf['std']['Vfwd'].values,
                                   n=gtdf['n']['Vfwd'].values,
                                   label=flymad_analysis.human_label(cgt),#laser_desc,
                                   order=100,
                                   color=flymad_plot.BLACK,
                                   N=len(gtdf['df']['obj_id'].unique()),
                                   df=gtdf['df'],
            )

        figure_title = "Moonwalker Thorax %s (%s)" % (gt,smoothstr)

        fig = plt.figure(figure_title, figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        l, axs, figs = flymad_plot.plot_timeseries_with_activation(ax,
                            targetbetween=dict(xaxis=gtdf['first']['t'].values,
                                               where=gtdf['first']['laser_state'].values>0),
                            sem=True,
                            downsample=5,
                            note="%s\n%s\n" % (gt,note),
                            individual={k:{'groupby':('obj_id','trial'),'xaxis':'t','yaxis':'Vfwd'} for k in datasets},
                            individual_title=figure_title + ' Individual Traces',
                            **datasets
        )

        ax.axhline(color='k', linestyle='--',alpha=0.8)
        ax.set_ylim(YLIM)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fwd Velocity (%s/s)' % arena.unit)
        ax.set_xlim(XLIM)

        flymad_plot.retick_relabel_axis(ax, XTICKS, YTICKS)

        fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s.png" % gt), bbox_inches='tight')
        fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s.svg" % gt), bbox_inches='tight')

        for efigname, efig in figs.iteritems():
            efig.savefig(flymad_plot.get_plotpath(path,"moonwalker_%s_individual_%s.png" % (gt, efigname)), bbox_inches='tight')

if __name__ == "__main__":

    DAY_1_GENOTYPES = ['50660chrim', '50660trp']
    DAY_1_CALIBRATION = 'calibration20140205_XXXXXX.filtered.yaml'

    DAY_2_GENOTYPES = ['50660','wtrp']
    DAY_2_CALIBRATION = 'calibration20140219_064948.filtered.yaml'

    LASER_POWERS = ['350ru','033ru','030ru','028ru',
                    '434iru','350iru','266iru','183iru']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--calibration-dir', help='calibration directory containing yaml files', required=True)

    args = parser.parse_args()
    path = args.path[0]

    medfilt = 51
    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    all_data = {k:dict() for k in DAY_1_GENOTYPES + DAY_2_GENOTYPES}

    #### BOTH DAYS EXPERIMENTS WERE RUN WITH DIFFERENT CALIBRATIONS
    #### DAY 1
    calibration_file = os.path.join(args.calibration_dir, DAY_1_CALIBRATION)
    d1_arena = madplot.Arena(
                'mm',
                **flymad_analysis.get_arena_conf(calibration_file=calibration_file))
    cache_fname = os.path.join(path,'moonwalker_d1.madplot-cache')
    cache_args = (path, DAY_1_GENOTYPES, LASER_POWERS, smoothstr, d1_arena)
    d1_data = None
    if args.only_plot:
        d1_data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if d1_data is None:
        #these loops are braindead inefficient and the wrong way,
        #however, we have limited time, and I cache the intermediate
        #representation anyway...
        d1_data = {k:dict() for k in DAY_1_GENOTYPES}
        for gt in DAY_1_GENOTYPES:
            for lp in LASER_POWERS:
                try:
                    d1_data[gt][lp] = prepare_data(path, d1_arena, args.smooth, medfilt, lp, [gt])
                except KeyError:
                    #this laser power and genotype combination was not tested
                    pass
        madplot.save_bagfile_cache(d1_data, cache_args, cache_fname)
    all_data.update(d1_data)
    #### DAY 2
    calibration_file = os.path.join(args.calibration_dir, DAY_2_CALIBRATION)
    d2_arena = madplot.Arena(
                'mm',
                **flymad_analysis.get_arena_conf(calibration_file=calibration_file))
    cache_fname = os.path.join(path,'moonwalker_d2.madplot-cache')
    cache_args = (path, DAY_2_GENOTYPES, LASER_POWERS, smoothstr, d2_arena)
    d2_data = None
    if args.only_plot:
        d2_data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if d2_data is None:
        #these loops are braindead inefficient and the wrong way,
        #however, we have limited time, and I cache the intermediate
        #representation anyway...
        d2_data = {k:dict() for k in DAY_2_GENOTYPES}
        for gt in DAY_2_GENOTYPES:
            for lp in LASER_POWERS:
                try:
                    d2_data[gt][lp] = prepare_data(path, d2_arena, args.smooth, medfilt, lp, [gt])
                except KeyError:
                    #this laser power and genotype combination was not tested
                    pass
        madplot.save_bagfile_cache(d2_data, cache_args, cache_fname)
    all_data.update(d2_data)

    note = "%s %s\nd1arena:%r\nd2arena:%r\nmedfilt %s" % (d1_arena.unit, smoothstr, d1_arena, d2_arena, medfilt)

    #from here on, arena is only used for the units
    assert d1_arena.unit == d2_arena.unit

    plot_all_data(path, all_data, d1_arena, note)
    plot_cross_activation_only(path, all_data, d1_arena, note)

    if args.show:
        plt.show()

