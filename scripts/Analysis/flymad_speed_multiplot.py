import argparse
import os
import sys
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset

import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import flymad_analysis

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.__version__ == "0.11.0"

def prepare_data(path, exp_genotype, ctrl_genotype):

    pooldf = DataFrame()
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

    df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).mean()
    df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).std()
    df2mean.to_csv((path + "/Speed_calculations/overlaid_means.csv"))
    df2std.to_csv((path + "/Speed_calculations/overlaid_std.csv"))

    #matplotlib seems sensitive to non-float colums, so convert to
    #float anything we plot
    for _df in [df2mean, df2std]:
        real_cols = [col for col in _df.columns if col not in ("Genotype", "lasergroup")]
        _df[ real_cols ] = _df[ real_cols ].astype(float)

    # half-assed, uninformed danno's tired method of grouping for plots:
    expmean = df2mean[df2mean['Genotype'] == exp_genotype]
    ctrlmean = df2mean[df2mean['Genotype']== ctrl_genotype]
    expstd = df2std[df2std['Genotype'] == exp_genotype]
    ctrlstd = df2std[df2std['Genotype']== ctrl_genotype]

    #save dataframes for faster replotting
    df2.save(path + "/df2.df")
    expmean.save(path + "expmean.df")
    ctrlmean.save(path + "ctrlmean.df")
    expstd.save(path + "expstd.df")
    ctrlstd.save(path + "ctrlstd.df")

    return expmean, ctrlmean, expstd, ctrlstd

def load_data( path ):
    return pd.load(path + "expmean.df"), pd.load(path + "ctrlmean.df"), pd.load(path + "expstd.df"), pd.load(path + "ctrlstd.df")

def plot_data( path, expmean, ctrlmean, expstd, ctrlstd ):

    fig2 = plt.figure()
    #velocity overlay:
    ax4 = fig2.add_subplot(1,1,1)
    ax4.plot(expmean['align'].values, expmean['v'].values, 'b-', zorder=11, lw=3)
    trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
    ax4.fill_between(expmean['align'].values, (expmean['v'] + expstd['v']).values, (expmean['v'] - expstd['v']).values, color='b', alpha=0.1, zorder=6)
    plt.axhline(y=0, color='k')
    ax4.fill_between(expmean['align'].values, 0, 1, where=expmean['laser_state'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
    ax4.plot(ctrlmean['align'].values, ctrlmean['v'].values, 'k-', zorder=10, lw=3)
    ax4.fill_between(ctrlmean['align'].values, (ctrlmean['v'] + ctrlstd['v']).values, (ctrlmean['v'] - ctrlstd['v']).values, color='k', alpha=0.1, zorder=5)
    ax4.fill_between(ctrlmean['align'].values, 0, 1, where=ctrlmean['laser_state'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (pixels/s) +/- STD')

    plt.savefig((path + "/Speed_calculations/overlaid_plot.png"))
    plt.savefig((path + "/Speed_calculations/overlaid_plot.svg"))


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shib'
    CTRL_GENOTYPE = 'wshib'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path, EXP_GENOTYPE, CTRL_GENOTYPE)

    plot_data(path, *data)

    plt.show()

