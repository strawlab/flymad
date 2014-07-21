# python csv/plot_scored_courtship.py /mnt/strawscience/data/FlyMAD/DROPBOX_AS_SUBMITTED/FlyMAD_resubmission/scored_data/Persistent_courtship/exemplary_performers/ --exp-genotype wGP --other-genotypes wtrpmyc,40347trpmyc,G323,40347 --only-trajectories 100L --calibration-file /mnt/strawscience/data/FlyMAD/DROPBOX_AS_SUBMITTED/FlyMAD_resubmission/scored_data/calibrations/calibration20140219_064948.filtered.yaml --show
# python csv/plot_scored_courtship.py /mnt/strawscience/data/FlyMAD/DROPBOX_AS_SUBMITTED/FlyMAD_resubmission/scored_data/Persistent_courtship/ --exp-genotype wGP --other-genotypes wtrpmyc,40347trpmyc,G323,40347

import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import glob
import subprocess
import cPickle as pickle
import sys

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mimg

import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import kruskal, mannwhitneyu

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import flymad.madplot as madplot

try:
    from strawlab_mpl.spines import spine_placer
except ImportError:
    print "ERROR: please install strawlab_mpl for nicer plots"
    def spine_placer(*args, **kwargs):
        pass

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0", "0.12.0")

STATS_NUM_BINS = 40

MINIPLOTS_XTICKS = [0,200,400]

DIRECTED_COURTING_DIST = 50

def prepare_data(path, resample_bin, min_experiment_duration, target_movie):
    data = {}

    gts = {}

    #PROCESS SCORE FILES:
    pooldf = pd.DataFrame()
    for df,metadata in flymad_analysis.load_courtship_csv(path):
        csvfilefn,experimentID,date,time,genotype,laser,repID = metadata

        genotype = genotype+"_"+laser
        gts[genotype] = True

        targets = flymad_analysis.get_targets(path, date, target_movie)
        assert len(targets) == 4
        targets = pd.DataFrame(targets)
        targets = (targets + 0.5).astype(int)

        #CALCULATE DISTANCE FROM TARGETs, KEEP MINIMUM AS dtarget
        if targets is not None:
            dist = pd.DataFrame.copy(df, deep=True)
            dist['x0'] = df['x'] - targets.ix[0,'x']
            dist['y0'] = df['y'] - targets.ix[0,'y']
            dist['x1'] = df['x'] - targets.ix[1,'x']
            dist['y1'] = df['y'] - targets.ix[1,'y']
            dist['x2'] = df['x'] - targets.ix[2,'x']
            dist['y2'] = df['y'] - targets.ix[2,'y']
            dist['x3'] = df['x'] - targets.ix[3,'x']
            dist['y3'] = df['y'] - targets.ix[3,'y']
            dist['d0'] = ((dist['x0'])**2 + (dist['y0'])**2)**0.5
            dist['d1'] = ((dist['x1'])**2 + (dist['y1'])**2)**0.5
            dist['d2'] = ((dist['x2'])**2 + (dist['y2'])**2)**0.5
            dist['d3'] = ((dist['x3'])**2 + (dist['y3'])**2)**0.5
            df['dtarget'] = dist.ix[:,'d0':'d3'].min(axis=1)               
        else:
            df['dtarget'] = 0

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < min_experiment_duration:
            print "\tmissing data? %s (duration %s < %s)" % (csvfilefn, duration, min_experiment_duration)
            continue

        print "\t%ss experiment" % duration

        #total_zx = df['zx'].sum()
        zx_series = df['zx']

        # do resampling
        df = df.resample(resample_bin, fill_method='ffill', how='median')
        zx_series_resampled = zx_series.resample( resample_bin, fill_method='ffill', how='sum')
        df['zx_counts'] = zx_series_resampled

        #fix laser_state due to resampling
        df['laser_state'] = df['laser_state'].fillna(value=0)
        df['laser_state'][df['laser_state'] > 0] = 1
        df['zx_binary'] = (df['zx'] > 0).values.astype(float)

        try:
            df = flymad_analysis.align_t_by_laser_on(
                    df,
                    resample_bin=resample_bin,
                    min_experiment_duration=min_experiment_duration,
                    align_first_only=True)
        except flymad_analysis.AlignError, err:
            print "\talign error %s (%s)" % (csvfilefn, err)
            continue

        #FIXME
        df['t_align'] = df['t']

        df['obj_id'] = flymad_analysis.create_object_id(date,time)
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        pooldf = pd.concat([pooldf, df])

    data = {}
    for gt in gts:
        gtdf = pooldf[pooldf['Genotype'] == gt]
        data[gt] = dict(df=gtdf)

    for gt in data:
        gtdf = data[gt]['df']

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

def plot_data(path, bin_size, dfs):

    figname = '_'.join(dfs)

    datasets = {}
    for gt in dfs:
        gtdf = dfs[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['zx'].values,
                            std=gtdf['std']['zx'].values,
                            n=gtdf['n']['zx'].values,
                            label=flymad_analysis.human_label(gt, specific=True),
                            order=flymad_analysis.get_genotype_order(gt),
                            df=gtdf['df'],
                            N=len(gtdf['df']['obj_id'].unique()))

    #assume protocol was identical with laser on
    ctrlmean = dfs[dfs.keys()[0]]['mean']

    figure_title = "Courtship Wingext"
    fig = plt.figure(figure_title)
    ax = fig.add_subplot(1,1,1)

    _,_,figs = flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=dict(xaxis=ctrlmean['t'].values,
                                       where=ctrlmean['laser_state'].values>0),
                    sem=True,
                    note="bin %s\n" % (bin_size,),
                    individual={k:{'groupby':'obj_id','xaxis':'t','yaxis':'zx'} for k in datasets},
                    individual_title=figure_title + ' Individual Traces',
                    **datasets
    )

    ax.set_xlabel('Time (s)')

    ax.set_ylabel('Wing extension index')
    ax.set_ylim([0,1])

    fig.savefig(flymad_plot.get_plotpath(path,"following_and_WingExt_%s.png" % figname), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"following_and_WingExt_%s.svg" % figname), bbox_inches='tight')

    for efigname, efig in figs.iteritems():
        efig.savefig(flymad_plot.get_plotpath(path,"following_and_WingExt_%s_individual_%s.png" % (figname, efigname)), bbox_inches='tight')

    datasets = {}
    for gt in dfs:
        gtdf = dfs[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['dtarget'].values,
                            std=gtdf['std']['dtarget'].values,
                            n=gtdf['n']['dtarget'].values,
                            label=flymad_analysis.human_label(gt, specific=True),
                            order=flymad_analysis.get_genotype_order(gt),
                            df=gtdf['df'],
                            N=len(gtdf['df']['obj_id'].unique()))

    #assume protocol was identical with laser on
    ctrlmean = dfs[dfs.keys()[0]]['mean']

    figure_title = "Courtship Dtarget"
    fig = plt.figure(figure_title)
    ax = fig.add_subplot(1,1,1)

    _,_,figs = flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=dict(xaxis=ctrlmean['t'].values,
                                       where=ctrlmean['laser_state'].values>0),
                    sem=True,
                    legend_location='lower right',
                    note="bin %s\n" % (bin_size,),
                    individual={k:{'groupby':'obj_id','xaxis':'t','yaxis':'dtarget'} for k in datasets},
                    individual_title=figure_title + ' Individual Traces',
                    **datasets
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (px)')

    fig.savefig(flymad_plot.get_plotpath(path,"following_and_dtarget_%s.png" % figname), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"following_and_dtarget_%s.svg" % figname), bbox_inches='tight')

    for efigname, efig in figs.iteritems():
        efig.savefig(flymad_plot.get_plotpath(path,"following_and_dtarget_%s_individual_%s.png" % (figname, efigname)), bbox_inches='tight')

if __name__ == "__main__":
    from plot_scored_courtship import plot_distance_histograms

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
#    parser.add_argument('--laser', default='140hpc', help='laser specifier')
    parser.add_argument('--calibration-file', help='calibration yaml files', required=False, default=None)
#    parser.add_argument('--exp-genotype', help='experimental genotype', required=True)
#    parser.add_argument('--other-genotypes', help='other genotypes (comma separated list)')
    parser.add_argument('--target-movie', help='path to mp4 that for indicating target locations FOR ALL CSV FILES GIVEN')
    parser.add_argument('--experiment-duration', type=int, default=600)

    args = parser.parse_args()
    path = args.path[0]

    if args.target_movie:
        if not os.path.isfile(args.target_movie):
            parser.error('could not find target movie %s' % args.target_movie)
    target_movie = args.target_movie

    min_experiment_duration = args.experiment_duration

#    gts = [args.exp_genotype] + (args.other_genotypes.split(',') if args.other_genotypes else [])

    bin_size = '1S'
    cache_fname = os.path.join(path,'courtship_%s.madplot-cache' % bin_size)
    cache_args = (bin_size,)
    dfs = None
    if args.only_plot:
        dfs = madplot.load_bagfile_cache(cache_args, cache_fname)
    if dfs is None:
        dfs = prepare_data(path, bin_size, min_experiment_duration, target_movie)
        madplot.save_bagfile_cache(dfs, cache_args, cache_fname)

    plot_data(path, bin_size, dfs)
    plot_distance_histograms(path, bin_size, dfs, one_plot_per_period=False)

    if args.show:
        plt.show()

