# python bag/plot_speed.py ~/Dropbox/140530-john_from_dan/OK371/bagfiles_with_flyinfo/*.bag --show --genotypes halo-ok371,db194-ok371,1101903-ok371,3008470-ok371,db132-ok371,haloctrl-ok371,1102110-ok371

import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import sys
import glob
import datetime
import os.path
import calendar

import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset

import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import flymad.madplot as madplot
import flymad.filename_regexes as filename_regexes

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0" ,  "0.12.0")

def prepare_data(bags, arena, smoothstr, smooth, medfilt, gts, min_experiment_duration):

    RESAMPLE_SPEC = '10L'

    found_gts = []

    pooldf = DataFrame()
    for bag in bags:
        df = madplot.load_bagfile_single_dataframe(bag, arena,
                                                        ffill=True,
                                                        smooth=smooth)
        df = flymad_analysis.resample(df, resample_specifier=RESAMPLE_SPEC)

        metadata = filename_regexes.parse_filename(bag, extract_genotype_and_laser=True)
        dateobj = filename_regexes.parse_date(bag)
        genotype = metadata['genotype'] + '-' + metadata['laser']

        found_gts.append(genotype)

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < min_experiment_duration:
            print "\tmissing data", bag
            continue

        print "\t%ss experiment" % duration

        #MAXIMUM SPEED = 300:
        #df['v'][df['v'] >= 300] = np.nan
        #df['v'] = df['v'].fillna(method='ffill')

        try:
            df = flymad_analysis.align_t_by_laser_on(
                    df, min_experiment_duration=min_experiment_duration,
                    align_first_only=True,
                    exact_num_ranges=1,
                    resample_bin=RESAMPLE_SPEC)
        except flymad_analysis.AlignError, err:
            print "\talign error %s (%s)" % (bag, err)
            pass

        #median filter
        if medfilt:
            df['v'] = scipy.signal.medfilt(df['v'].values, medfilt)

        df['obj_id'] = calendar.timegm(dateobj)
        df['Genotype'] = genotype

        pooldf = pd.concat([pooldf, df])

    data = {}
    for gt in set(found_gts):
        gtdf = pooldf[pooldf['Genotype'] == gt]

        grouped = gtdf.groupby(['t'], as_index=False)
        data[gt] = dict(mean=grouped.mean().astype(float),
                        std=grouped.std().astype(float),
                        n=grouped.count().astype(float),
                        first=grouped.first(),
                        df=gtdf)

    return data

def get_stats(group):
    return {'mean': group.mean(),
            'var' : group.var(),
            'n' : group.count()
           }

def plot_data(path, data, arena, note):

    #customize plot colors here. key is genotype name
    LABELS = {}
    COLORS = {}
    ORDERS = {}


    fig2 = plt.figure("Speed")
    ax = fig2.add_subplot(1,1,1)

    datasets = {}
    for gt in data:
        gtdf = data[gt]
        try:
            datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                                value=gtdf['mean']['v'].values,
                                std=gtdf['std']['v'].values,
                                n=gtdf['n']['v'].values,
                                label=LABELS.get(gt,gt),
                                color=COLORS.get(gt),
                                order=ORDERS.get(gt),
                                df=gtdf['df'],
                                N=len(gtdf['df']['obj_id'].unique()))
        except KeyError:
            print "MISSING DATA FOR",gt

    #assume laser on protocol was identical in all trials
    ctrlfirst = data[data.keys()[0]]['first']

    result_d = flymad_plot.plot_timeseries_with_activation(ax,
                targetbetween=dict(xaxis=ctrlfirst['t'].values,
                                   where=ctrlfirst['laser_state'].values>0),
                downsample=5,
                sem=True,
                note="OK371shits\n%s\n" % note,
                individual={k:{'groupby':'obj_id','xaxis':'t','yaxis':'v'} for k in datasets},
                individual_title='Speed Individual Traces',
                return_dict=True,
                **datasets
    )
    figs = result_d['figs']

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (%s/s)' % arena.unit)

#    ax.set_xlim([-15, 30])
#    ax.set_ylim([0, 18])
#    flymad_plot.retick_relabel_axis(ax, [-15, 0, 15, 30], [0, 5, 10, 15])

    fig2.savefig(flymad_plot.get_plotpath(path,"speed_plot.png"), bbox_inches='tight')
    fig2.savefig(flymad_plot.get_plotpath(path,"speed_plot.svg"), bbox_inches='tight')

    for efigname, efig in figs.iteritems():
        efig.savefig(flymad_plot.get_plotpath(path,"speed_plot_individual_%s.png" % efigname), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', help='path to bag files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--only-plot', action='store_true', default=False,
                        help='dont attempt to regenerate data. just replot')
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--genotypes', required=True,
                        help='comma separated list of genotypes to plot')
    parser.add_argument('--median-filter', default=51, type=int,
                        help='apply a median filter of this many samples to trajectories')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='perform pairwise comparison and show in plotly')
    parser.add_argument('--min-experiment-duration', default=70, type=float,
                        help='minimum experiment duration for a bag file to be considered valid. '\
                             'should be about 10 seconds shorter than the actual experiment')


    args = parser.parse_args()
    path = args.path[0]

    bags = args.path
    genotypes = args.genotypes.split(',')

    medfilt = args.median_filter
    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    arena = madplot.Arena('mm')

    note = "%s %s\n%r\nmedfilt %s" % (arena.unit, smoothstr, arena, medfilt)

    cache_fname = os.path.join(os.path.dirname(path),'speed.madplot-cache')
    cache_args = ([os.path.basename(b) for b in bags], smoothstr, args.smooth, medfilt, genotypes, args.min_experiment_duration)
    data = None
    if args.only_plot:
        data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if data is None:
        data = prepare_data(bags, arena, smoothstr, args.smooth, medfilt, genotypes, args.min_experiment_duration)
        madplot.save_bagfile_cache(data, cache_args, cache_fname)

    if args.stats:
        fname_prefix = flymad_plot.get_plotpath(path,'csv_speed')
        madplot.view_pairwise_stats_plotly(data, genotypes,
                                           fname_prefix,
                                           align_colname='t',
                                           stat_colname='v',
                                           )

    plot_data(path, data, arena, note)

    if args.show:
        plt.show()

