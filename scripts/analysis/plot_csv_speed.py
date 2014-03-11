import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset

import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import madplot
from scipy.stats import kruskal

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0" ,  "0.12.0")

EXPERIMENT_DURATION = 70.0

def prepare_data(path, arena, smoothstr, smooth, gts):

    pooldf = DataFrame()
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        cache_args = os.path.basename(csvfile), arena, smoothstr
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

        #we plot head v thorax v nolaser (so for the same of plotting, consider
        #these the genotypes
        genotype = genotype + '-' + laser

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        if 0:
            fig = plt.figure()
            fig.suptitle(os.path.basename(csvfile))
            ax = fig.add_subplot(1,1,1)
            df['experiment'] = 1
            df['tobj_id'] = 1
            madplot.plot_tracked_trajectory(ax, df, arena,
                        debug_plot=False,
                        color='k',
            )
            ax.add_patch(arena.get_patch(color='k', alpha=0.1))

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < EXPERIMENT_DURATION:
            print "\tmissing data", csvfilefn
            continue

        print "\t%ss experiment" % duration

        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 50] = np.nan
        df['v'] = df['v'].fillna(method='ffill')

        #Here we have a 10ms resampled dataframe at least EXPERIMENT_DURATION seconds long.
        df = df.head(flymad_analysis.get_num_rows(EXPERIMENT_DURATION))
        tb = flymad_analysis.get_resampled_timebase(EXPERIMENT_DURATION)
        #find when the laser first came on (argmax returns the first true value if
        #all values are identical
        t0idx = np.argmax(np.gradient(df['laser_state'].values > 0))
        t0 = tb[t0idx]
        df['t'] = tb - t0

        #groupby on float times is slow. make a special align column 
        df['t_align'] = np.array(range(0,len(df))) - t0idx

        df['obj_id'] = flymad_analysis.create_object_id(date,time)
        df['Genotype'] = genotype
        df['lasergroup'] = laser

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

def get_stats(group):
    return {'mean': group.mean(),
            'var' : group.var(),
            'n' : group.count()
           }

def add_obj_id(df):
    results = np.zeros( (len(df),), dtype=np.int )
    obj_id = 0
    for i,(ix,row) in enumerate(df.iterrows()):
        if row['align']==0.0:
            obj_id += 1
        results[i] = obj_id
    df['obj_id']=results
    return df

def calc_kruskal(df_ctrl, df_exp, number_of_bins, align_colname='align', vfwd_colname='v'):
    df_ctrl = add_obj_id(df_ctrl)
    df_exp = add_obj_id(df_exp)

    dalign = df_ctrl['align'].max() - df_ctrl['align'].min()

    p_values = DataFrame()
    for binsize in number_of_bins:
        bins = np.linspace(0,dalign,binsize)
        binned_ctrl = pd.cut(df_ctrl['align'], bins, labels= bins[:-1])
        binned_exp = pd.cut(df_exp['align'], bins, labels= bins[:-1])
        for x in binned_ctrl.levels:
            test1_all_flies_df = df_ctrl[binned_ctrl == x]
            test1 = []
            for obj_id, fly_group in test1_all_flies_df.groupby('obj_id'):
                test1.append( np.mean(fly_group['v'].values) )
            test1 = np.array(test1)

            test2_all_flies_df = df_exp[binned_exp == x]
            test2 = []
            for obj_id, fly_group in test2_all_flies_df.groupby('obj_id'):
                test2.append( np.mean(fly_group['v'].values) )
            test2 = np.array(test2)

            hval, pval = kruskal(test1, test2)
            dftemp = DataFrame({'Total_bins': binsize , 'Bin_number': x, 'P': pval}, index=[x])
            p_values = pd.concat([p_values, dftemp])
    return p_values

def run_stats (path, arena, exp_genotype, ctrl_genotype, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln , df2):
    number_of_bins = [ 6990//4 ]
    df_ctrl = df2[df2['Genotype'] == ctrl_genotype]
    df_exp = df2[df2['Genotype'] == exp_genotype]
    return calc_kruskal(df_ctrl, df_exp, number_of_bins)

def fit_to_curve (path, arena, smoothstr, p_values):
    x = np.array(p_values['Bin_number'][p_values['Bin_number'] <= 50])
    logs = -1*(np.log(p_values['P'][p_values['Bin_number'] <= 50]))
    y = np.array(logs)
    # order = 11 #DEFINE ORDER OF POLYNOMIAL HERE.
    # poly_params = np.polyfit(x,y,order)
    # polynom = np.poly1d(poly_params)
    # xPoly = np.linspace(0, max(x), 100)
    # yPoly = polynom(xPoly)
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ax.plot(x, y, 'bo-')

    ax.axvspan(10,20,
               facecolor='Yellow', alpha=0.15,
               edgecolor='none',
               zorder=-20)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('-log(p)')
    ax.set_ylim([0, 25])
    ax.set_xlim([5, 40])

def plot_data(path, data, arena, note):

    LABELS = {'OK371shits-130h':'OK371>ShibireTS (head)',
              'OK371shits-nolaser':'Control',
              'OK371shits-130t':'OK371>ShibireTS (thorax)',
    }

    COLORS = {'OK371shits-130h':flymad_plot.RED,
              'OK371shits-nolaser':flymad_plot.BLACK,
              'OK371shits-130t':flymad_plot.ORANGE,
    }

    fig2 = plt.figure("Speed")
    ax = fig2.add_subplot(1,1,1)

    datasets = {}
    for gt in data:
        gtdf = data[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['v'].values,
                            std=gtdf['std']['v'].values,
                            n=gtdf['n']['v'].values,
                            label=LABELS[gt],
                            color=COLORS[gt],
                            N=len(gtdf['df']['obj_id'].unique()))

    ctrlmean = data['OK371shits-nolaser']['mean']

    flymad_plot.plot_timeseries_with_activation(ax,
                targetbetween=dict(xaxis=ctrlmean['t'].values,
                                   where=ctrlmean['laser_state'].values>0),
                downsample=25,
                sem=True,
                note="OK371shits\n%s\n" % note,
                **datasets
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (%s/s)' % arena.unit)
#    ax.set_xlim([0, 50])

#    ax.set_title("Speed")

    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.png"), bbox_inches='tight')
    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.svg"), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shits-130h'
    CTRL_GENOTYPE = 'OK371shits-nolaser'
    EXP2_GENOTYPE = 'OK371shits-130t'
    CALIBRATION_FILE = 'calibration20140219_064948.filtered.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--calibration-dir', help='calibration directory containing yaml files', required=True)

    args = parser.parse_args()
    path = args.path[0]

    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    calibration_file = os.path.join(args.calibration_dir, CALIBRATION_FILE)
    arena = madplot.Arena(
                'mm',
                **flymad_analysis.get_arena_conf(calibration_file=calibration_file))

    note = "%s %s\n%r" % (arena.unit, smoothstr, arena)

    data = prepare_data(path, arena, smoothstr, args.smooth, [EXP_GENOTYPE, CTRL_GENOTYPE, EXP2_GENOTYPE])

    #p_values = run_stats(path, arena, EXP_GENOTYPE, CTRL_GENOTYPE, *data)
    #fit_to_curve(path, arena, smoothstr, p_values)
    plot_data(path, data, arena, note)

    if args.show:
        plt.show()

