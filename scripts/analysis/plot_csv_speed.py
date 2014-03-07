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

RESAMPLE_SPECIFIER = '10L'

def _load_and_smooth_csv(csvfile, arena, smooth, resample_specifier):
    csvfilefn = os.path.basename(csvfile)
    try:
        experimentID,date,time = csvfilefn.split("_",2)
        genotype,laser,repID = experimentID.split("-",2)
        repID = repID + "_" + date
        print "processing: ", experimentID
    except:
        print "invalid filename:", csvfilefn
        return None

    df = pd.read_csv(csvfile, index_col=0)

    if not df.index.is_unique:
        raise Exception("CORRUPT CSV. INDEX (NANOSECONDS SINCE EPOCH) MUST BE UNIQUE")

    #resample to 10ms (mean) and set a proper time index on the df
    df = flymad_analysis.fixup_index_and_resample(df, resample_specifier)

    #smooth the positions, and recalculate the velocitys based on this.
    dt = flymad_analysis.kalman_smooth_dataframe(df, arena, smooth)

    return df,dt,experimentID,date,time,genotype,laser,repID

def prepare_data(path, arena, smoothstr, smooth, exp_genotype, ctrl_genotype, exp2_genotype):

    df2 = DataFrame()
    if not os.path.exists(path + "/Speed_calculations/"):
        os.makedirs(path + "/Speed_calculations/")
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        cache_args = os.path.basename(csvfile), arena, smoothstr, RESAMPLE_SPECIFIER
        cache_fname = csvfile+'.madplot-cache'

        results = madplot.load_bagfile_cache(cache_args, cache_fname)
        if results is None:
            results = _load_and_smooth_csv(csvfile, arena, smooth, RESAMPLE_SPECIFIER)
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


        df['Genotype'] = genotype

        df['laser_state'] = df['laser_state'].fillna(value=0)

        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 50] = np.nan
        df['v'] = df['v'].fillna(method='ffill')

        #find the time of the first laseron. iterating is sloww and ugly, but meh
        before = after = None
        for idx,rowdf in df.iterrows():
            if rowdf['laser_state'] > 0:
                before = idx - DateOffset(seconds=20)
                after = idx + DateOffset(seconds=30)
                break

        assert before != None
        assert (after - before).total_seconds() == 50

        dftemp = df[before:after][['Genotype', 'v', 'laser_state']]
        dftemp['align'] = np.linspace(0,(after-before).total_seconds(),len(dftemp))

        assert len(dftemp) == 5001

        df2 = pd.concat([df2, dftemp])

    expdf = df2[df2['Genotype'] == exp_genotype]
    ctrldf = df2[df2['Genotype']== ctrl_genotype]
    exp2df = df2[df2['Genotype']== exp2_genotype]

    expmean = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlmean = ctrldf.groupby(['align'], as_index=False).mean().astype(float)
    exp2mean = exp2df.groupby(['align'], as_index=False).mean().astype(float)

    expstd = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlstd = ctrldf.groupby(['align'], as_index=False).mean().astype(float)
    exp2std = exp2df.groupby(['align'], as_index=False).mean().astype(float)

    expn = expdf.groupby(['align'], as_index=False).count().astype(float)
    ctrln = ctrldf.groupby(['align'], as_index=False).count().astype(float)
    exp2n = exp2df.groupby(['align'], as_index=False).count().astype(float)

    return expmean, ctrlmean, exp2mean, expstd, ctrlstd, exp2std, expn, ctrln, exp2n, df2

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

def plot_data(path, arena, smoothstr, dfs):

    expmean, ctrlmean, exp2mean, expstd, ctrlstd, exp2std, expn, ctrln, exp2n, df2 = dfs

    fig2 = plt.figure("Speed Multiplot %s" % smoothstr)
    ax = fig2.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    exp=dict(xaxis=expmean['align'].values,
                             value=expmean['v'].values,
                             std=expstd['v'].values,
                             n=expn['v'].values,
                             label='OK371>ShibireTS (head)',
                             ontop=True),
                    ctrl=dict(xaxis=ctrlmean['align'].values,
                              value=ctrlmean['v'].values,
                              std=ctrlstd['v'].values,
                              n=ctrln['v'].values,
                              label='Control'),
                    exp2=dict(xaxis=exp2mean['align'].values,
                             value=exp2mean['v'].values,
                             std=exp2std['v'].values,
                             n=exp2n['v'].values,
                             label='OK371>ShibireTS (body)'),
                    targetbetween=ctrlmean['laser_state'].values>0,
                    downsample=25,
                    sem=True
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (%s/s) +/- STD' % arena.unit)
    ax.set_xlim([0, 50])

    ax.set_title("Speed %s" % smoothstr)

    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot_%s.png" % smoothstr), bbox_inches='tight')
    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot_%s.svg" % smoothstr), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shits-130h'
    CTRL_GENOTYPE = 'OK371shits-nolaser'
    EXP2_GENOTYPE = 'OK371shits-130t'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--calibration', default=None, help='calibration yaml file')

    args = parser.parse_args()
    path = args.path[0]

    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    arena = madplot.Arena(
                'mm',
                **flymad_analysis.get_arena_conf(calibration_file=args.calibration))

    data = prepare_data(path, arena, smoothstr, args.smooth, EXP_GENOTYPE, CTRL_GENOTYPE, EXP2_GENOTYPE)

    #p_values = run_stats(path, arena, EXP_GENOTYPE, CTRL_GENOTYPE, *data)
    #fit_to_curve(path, arena, smoothstr, p_values)
    plot_data(path, arena, smoothstr, data)

    if args.show:
        plt.show()

