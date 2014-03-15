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

EXPERIMENT_DURATION = 80.0

YLIM = [-10, 25]
YTICKS = [-20, 0, 20, 40]

XLIM = [-10, 60]
XTICKS = [0, 30, 60]

TS_FIGSIZE = (10,6)

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
            print "\tmissing data", csvfilefn, duration
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

        #Here we have a 10ms resampled dataframe at least EXPERIMENT_DURATION seconds long.
        df = df.head(flymad_analysis.get_num_rows(EXPERIMENT_DURATION))
        tb = flymad_analysis.get_resampled_timebase(EXPERIMENT_DURATION)
        #find when the laser first came on (argmax returns the first true value if
        #all values are identical
        dlaser = np.gradient( (df['laser_state'].values > 0).astype(int) ) > 0
        t0idx = np.argmax(dlaser)
        t0 = tb[t0idx-1]
        df['t'] = tb - t0

        #groupby on float times is slow. make a special align column
        df['t_align'] = np.array(range(0,len(df))) - t0idx

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

        grouped = gtdf.groupby(['t_align'], as_index=False)

        data[gt] = dict(mean=grouped.mean().astype(float),
                        std=grouped.std().astype(float),
                        n=grouped.count().astype(float),
                        first=grouped.first(),
                        df=gtdf)

    return data

def plot_all_data(path, data, arena, note, laser='350iru'):

    ORDER = ["50660trp","50660","wtrp"]
    COLORS = {"50660trp":flymad_plot.RED,"50660":flymad_plot.BLACK,"wtrp":flymad_plot.BLUE}

    datasets = {}

    figure_title = "Moonwalker"
    fig = plt.figure(figure_title, figsize=TS_FIGSIZE)
    ax = fig.add_subplot(1,1,1)

    for gt in data:
        gtdf = data[gt][laser][gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['Vfwd'].values,
                            std=gtdf['std']['Vfwd'].values,
                            n=gtdf['n']['Vfwd'].values,
                            order=ORDER.index(gt),
                            label=flymad_analysis.human_label(gt),
                            color=COLORS[gt],
                            N=len(gtdf['df']['obj_id'].unique()),
                            df=gtdf['df'],
        )

    l, axs, figs = flymad_plot.plot_timeseries_with_activation(ax,
                        targetbetween=dict(xaxis=data['50660trp'][laser]['50660trp']['first']['t'].values,
                                           where=data['50660trp'][laser]['50660trp']['first']['laser_state'].values>0),
                        downsample=25,
                        note="%s\n" % (note,),
                        individual={k:{'groupby':'obj_id','xaxis':'t','yaxis':'Vfwd'} for k in datasets},
                        individual_title=figure_title + ' Individual Traces',
                        **datasets
    )

    ax.axhline(color='k', linestyle='--',alpha=0.8)
    ax.set_ylim(YLIM)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Forward velocity (%s/s)' % arena.unit)
    ax.set_xlim(XLIM)

    flymad_plot.retick_relabel_axis(ax, XTICKS, YTICKS)

    fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_hate.png"), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"moonwalker_hate.svg"), bbox_inches='tight')


if __name__ == "__main__":

    DAY_1_GENOTYPES = ['50660', '50660trp', 'wtrp']

    LASER_POWERS = ['350iru']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--calibration-dir', help='calibration directory containing yaml files', required=False)

    args = parser.parse_args()
    path = args.path[0]

    medfilt = 51
    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    calibration_file = None
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
    all_data = d1_data

    note = "%s %s\nd1arena:%r\nmedfilt %s" % (d1_arena.unit, smoothstr, d1_arena, medfilt)
    plot_all_data(path, all_data, d1_arena, note)

    if args.show:
        plt.show()

