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

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.__version__ == "0.11.0"

arena = flymad_analysis.Arena('mm')

def prepare_data(path, smoothstr, smooth, exp_genotype, ctrl_genotype):

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
        dt = flymad_analysis.kalman_smooth_dataframe(df, arena, smooth)

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
    df2.save(path + "/df2" + smoothstr + ".df")
    expmean.save(path + "/expmean" + smoothstr + ".df")
    ctrlmean.save(path + "/ctrlmean" + smoothstr + ".df")
    expstd.save(path + "/expstd" + smoothstr + ".df")
    ctrlstd.save(path + "/ctrlstd" + smoothstr + ".df")
    expn.save(path + "/expn" + smoothstr + ".df")
    ctrln.save(path + "/ctrln" + smoothstr + ".df")

    return expmean, ctrlmean, expstd, ctrlstd, expn, ctrln

def load_data(path, smoothstr):
    return (
            pd.load(path + "/expmean" + smoothstr + ".df"),
            pd.load(path + "/ctrlmean" + smoothstr + ".df"),
            pd.load(path + "/expstd" + smoothstr + ".df"),
            pd.load(path + "/ctrlstd" + smoothstr + ".df"),
            pd.load(path + "/expn" + smoothstr + ".df"),
            pd.load(path + "/ctrln" + smoothstr + ".df"),
    )

def plot_data( path, smoothstr, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln ):

    fig2 = plt.figure("Speed Multiplot %s" % smoothstr)
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
    ax.set_ylabel('Speed (%s/s) +/- STD' % arena.unit)
    ax.set_xlim([0, 70])

    ax.set_title("Speed %s" % smoothstr)

    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot_%s.png" % smoothstr), bbox_inches='tight')
    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot_%s.svg" % smoothstr), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shib'
    CTRL_GENOTYPE = 'wshib'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)

    args = parser.parse_args()
    path = args.path[0]

    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    print "speed", smoothstr

    if args.only_plot:
        data = load_data(path,smoothstr)
    else:
        data = prepare_data(path, smoothstr, args.smooth, EXP_GENOTYPE, CTRL_GENOTYPE)

    plot_data(path, smoothstr, *data)

    if args.show:
        plt.show()


