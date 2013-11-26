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
from scipy.stats import kruskal

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0" ,  "0.12.0")

def prepare_data(path, exp_genotype, ctrl_genotype):

    df2 = DataFrame()
    if not os.path.exists(path + "/Speed_calculations/"):
        os.makedirs(path + "/Speed_calculations/")
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        csvfilefn = os.path.basename(csvfile)
        try:
            experimentID,date,time = csvfilefn.split("_",2)
            genotype,laser,repID = experimentID.split("-",2)
            repID = repID + "_" + date
            print "processing: ", experimentID
        except:
            print "invalid filename:", csvfilefn
            continue 
        df = pd.read_csv(csvfile, index_col=0)

        if not df.index.is_unique:
            raise Exception("CORRUPT CSV. INDEX (NANOSECONDS SINCE EPOCH) MUST BE UNIQUE")


        #resample to 10ms (mean) and set a proper time index on the df
        df = flymad_analysis.fixup_index_and_resample(df, '10L')

        #smooth the positions, and recalculate the velocitys based on this.
        dt = flymad_analysis.kalman_smooth_dataframe(df)

        df['laser_state'] = df['laser_state'].fillna(value=0)

        lasermask = df[df['laser_state'] == 1]
        df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)

        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 300] = np.nan
        
        #the resampling above, using the default rule of 'mean' will, if the laser
        #was on any time in that bin, increase the mean > 0.
        df['laser_state'][df['laser_state'] > 0] = 1
            
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        #combine 60s trials together into df2:
        dfshift = df.shift()
        laserons = df[ (df['laser_state']-dfshift['laser_state']) == 1 ]    
        #SILLY BUG. Sometimes laserons contains incorrect timepoints. To fix this, I compare each laseron to the 
        # previous and discard if ix - prev is less than the experimental looping time.
        prev = pd.DatetimeIndex([datetime.datetime(1986, 5, 27)])[0]  #set to random time initially
        for ix,row in laserons.iterrows():
            before = ix - DateOffset(seconds=9.95)
            after = ix + DateOffset(seconds=59.95)
            if (ix - prev).total_seconds() <= 59.95:
                continue
            else:
                print "ix:", ix, "\t span:", ix - prev
                prev = ix
                dftemp = df.ix[before:after][['Genotype', 'lasergroup','v', 'laser_state']]
                dftemp['align'] = np.linspace(0,(after-before).total_seconds(),len(dftemp))
                df2 = pd.concat([df2, dftemp])

    expdf = df2[df2['Genotype'] == exp_genotype]
    ctrldf = df2[df2['Genotype']== ctrl_genotype]

    #we no longer need to group by genotype, and lasergroup is always the same here
    #so just drop it. 
    assert len(expdf['lasergroup'].unique()) == 1, "only one lasergroup handled"

    expmean = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlmean = ctrldf.groupby(['align'], as_index=False).mean().astype(float)

    expstd = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlstd = ctrldf.groupby(['align'], as_index=False).mean().astype(float)

    expn = expdf.groupby(['align'], as_index=False).count().astype(float)
    ctrln = ctrldf.groupby(['align'], as_index=False).count().astype(float)

    ####AAAAAAAARRRRRRRRRRRGGGGGGGGGGGGGGHHHHHHHHHH so much copy paste here
    df2.save(path + "/df2.df")
    expmean.save(path + "/expmean.df")
    ctrlmean.save(path + "/ctrlmean.df")
    expstd.save(path + "/expstd.df")
    ctrlstd.save(path + "/ctrlstd.df")
    expn.save(path + "/expn.df")
    ctrln.save(path + "/ctrln.df")

    return expmean, ctrlmean, expstd, ctrlstd, expn, ctrln, df2

def load_data( path ):
    return (
            pd.read_pickle(path + "/df2.df"),
            pd.read_pickle(path + "/expmean.df"),
            pd.read_pickle(path + "/ctrlmean.df"),
            pd.read_pickle(path + "/expstd.df"),
            pd.read_pickle(path + "/ctrlstd.df"),
            pd.read_pickle(path + "/expn.df"),
            pd.read_pickle(path + "/ctrln.df"),
    )

def get_stats(group):
    return {'mean': group.mean(),
            'var' : group.var(),
            'n' : group.count()
           }

def run_stats (path, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln , df2): 
    print type(df2), df2.shape 
    number_of_bins = [ 6990 ] #69.9 second trials.
    p_values = DataFrame()  
    df_ctrl = df2[df2['Genotype'] == 'wshib']
    df_exp = df2[df2['Genotype'] == 'ok371shib']
    for binsize in number_of_bins:
        bins = np.linspace(0,8.91, binsize)
        binned_ctrl = pd.cut(df_ctrl['align'], bins, labels= bins[:-1])
        binned_exp = pd.cut(df_exp['align'], bins, labels= bins[:-1])
        for x in binned_ctrl.levels:                
            test1 = df_ctrl['v'][binned_ctrl == x]
            test2 = df_exp['v'][binned_exp == x]
            hval, pval = kruskal(test1, test2)
            dftemp = DataFrame({'Total_bins': binsize , 'Bin_number': x, 'P': pval}, index=[x])
            p_values = pd.concat([p_values, dftemp])
    return p_values

def plot_data( path, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln ):

    fig2 = plt.figure("Speed Multiplot")
    ax = fig2.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    exp=dict(xaxis=expmean['align'].values,
                             value=expmean['v'].values,
                             std=expstd['v'].values,
                             n=expn['v'].values,
                             label='OK371>ShibireTS',
                             ontop=True),
                    ctrl=dict(xaxis=ctrlmean['align'].values,
                              value=ctrlmean['v'].values,
                              std=ctrlstd['v'].values,
                              n=ctrln['v'].values,
                              label='Control'),
                    targetbetween=ctrlmean['laser_state'].values>0,
                    downsample=25
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (pixels/s) +/- STD')
    ax.set_ylim([0, 160])
    ax.set_xlim([0, 70])

    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.png"), bbox_inches='tight')
    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.svg"), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shib'
    CTRL_GENOTYPE = 'wshib'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path, EXP_GENOTYPE, CTRL_GENOTYPE)

    run_stats(path, *data)
    p_values.to_csv(path + '/p_values.csv')
    plot_data(path, *data)

    if args.show:
        plt.show()


