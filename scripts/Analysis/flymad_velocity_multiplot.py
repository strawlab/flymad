import argparse
import math
import glob
import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import flymad_analysis

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.__version__ == "0.11.0"

def prepare_data(path, exp_genotype, ctrl_genotype):

    pooldf = DataFrame()
    df2 = DataFrame()
    if not os.path.exists(path + "/Velocity_calculations/"):
        os.makedirs(path + "/Velocity_calculations/")
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        csvfilefn = os.path.basename(csvfile)
        try:
            experimentID,date,_time = csvfilefn.split("_",2)
            genotype,laser,repID = experimentID.split("-",2)
            repID = repID + "_" + date
            print "processing: ", experimentID
        except:
            print "invalid filename:", csvfilefn
            continue 
        df = pd.read_csv(csvfile, index_col=0)

        if not df.index.is_unique:
            raise Exception("CORRUPT CSV. INDEX (NANOSECONDS SINCE EPOCH) MUST BE UNIQUE")

        #pandas doesn't know how to resample columns with type 'object'. Which is
        #fair. so fix up the scoring colums, and their meanings now
        #ROTATE by pi if orientation is east
        flymad_analysis.fix_scoring_colums(df,
                        valmap={'zx':{'z':math.pi,'x':0},
                                'as':{'a':1,'s':0}})

        #resample to 10ms (mean) and set a proper time index on the df
        df = flymad_analysis.fixup_index_and_resample(df, '10L')

        #smooth the positions, and recalculate the velocitys based on this.
        dt = flymad_analysis.kalman_smooth_dataframe(df)

        df['laser_state'] = df['laser_state'].fillna(value=0)

        lasermask = df[df['laser_state'] == 1]
        df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)

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

        #the resampling above, using the default rule of 'mean' will, if the laser
        #was on any time in that bin, increase the mean > 0.
        df['laser_state'][df['laser_state'] > 0] = 1
            
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID
        
        #combine 7s trials together into df2:
        dfshift = df.shift()
        laserons = df[ (df['laser_state']-dfshift['laser_state']) == 1 ]

        prev = pd.DatetimeIndex([datetime.datetime(1986, 5, 27)])[0]
        for ix,row in laserons.iterrows():
            before = ix - DateOffset(seconds=1.95)
            after = ix + DateOffset(seconds=6.95)
            if (ix - prev).total_seconds() <= 6.95:
                continue
            else:
                print "ix:", ix, "\t span:", ix - prev
                prev = ix
                dftemp = df.ix[before:after][['Genotype', 'lasergroup','Vfwd', 'Afwd', 'dorientation', 'laser_state']]
                dftemp['align'] = np.linspace(0,(after-before).total_seconds(),len(dftemp))
                df2 = pd.concat([df2, dftemp])

    #    pooldf = pd.concat([pooldf, df])   

    #    df.to_csv((path + "/Velocity_calculations/" + experimentID + "_" + date + ".csv"), index=False)
    print "DF2:", df2.columns

    """
    means = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False)[['Vfwd', 'Afwd', 'dorientation']].mean()
    stds = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False)[['Vfwd', 'Afwd', 'dorientation']].std()
    means.to_csv((path + "/Velocity_calculations/means.csv"))
    """

    df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].mean()
    df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].std()
    df2mean.to_csv((path + "/Velocity_calculations/overlaid_means.csv"))
    df2std.to_csv((path + "/Velocity_calculations/overlaid_std.csv"))

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
    expmean.save(path + "/expmean.df")
    ctrlmean.save(path + "/ctrlmean.df")
    expstd.save(path + "/expstd.df")
    ctrlstd.save(path + "/ctrlstd.df")

    return expmean, ctrlmean, expstd, ctrlstd

def load_data( path ):
    return pd.load(path + "/expmean.df"), pd.load(path + "/ctrlmean.df"), pd.load(path + "/expstd.df"), pd.load(path + "/ctrlstd.df")

def plot_data( path, expmean, ctrlmean, expstd, ctrlstd ):

    # PLOT
    """
    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax.plot(means['tracked_t'].values, means['Vfwd'].values, 'b-', zorder=1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(means['tracked_t'].values, 0, 1, where=(means['laser_power']>0.9).values, facecolor='Yellow', alpha=0.3, transform=trans)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fwd Velocity (pixels/s)')
    #ax.set_xlim([-10,70])
    plt.axhline(y=0, color='k')

    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(means['tracked_t'].values, means['Afwd'].values, 'r-')
    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.fill_between(means['tracked_t'].values, 0, 1, where=(means['laser_power']>0.9).values, facecolor='Yellow', alpha=0.3, transform=trans)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Fwd Acceleration (pixels/s^2)')
    #ax2.set_xlim([-10,70])
    plt.axhline(y=0, color='k')

    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(means['tracked_t'].values, means['dorientation'].values, 'k-')
    trans = mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
    ax3.fill_between(means['tracked_t'].values, 0, 1, where=(meanas['laser_power']>0.9).values, facecolor='Yellow', alpha=0.3, transform=trans)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation (Radians/s)')
    #ax3.set_xlim([-10,70])


    plt.savefig((path + "/Velocity_calculations/singletrace_plot.png"))
    """
    #PLOT OVERLAY

    fig2 = plt.figure()
    #velocity overlay:
    ax4 = fig2.add_subplot(1,1,1)
    ax4.plot(expmean['align'].values, expmean['Vfwd'].values, 'b-', zorder=10, lw=3)
    trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
    ax4.fill_between(expmean['align'].values, (expmean['Vfwd'] + expstd['Vfwd']).values, (expmean['Vfwd'] - expstd['Vfwd']).values, color='b', alpha=0.1, zorder=5)
    plt.axhline(y=0, color='k')
    ax4.fill_between(expmean['align'].values, 0, 1, where=expmean['laser_state'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
    ax4.plot(ctrlmean['align'].values, ctrlmean['Vfwd'].values, 'k-', zorder=11, lw=3)
    ax4.fill_between(ctrlmean['align'].values, (ctrlmean['Vfwd'] + ctrlstd['Vfwd']).values, (ctrlmean['Vfwd'] - ctrlstd['Vfwd']).values, color='k', alpha=0.1, zorder=6)
    ax4.fill_between(ctrlmean['align'].values, 0, 1, where=ctrlmean['laser_state'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Fwd Velocity (pixels/s) +/- STD')
    """
    #acceleration overlay:
    ax5 = fig2.add_subplot(3,1,2)
    ax5.plot(df2mean['align'].values, df2mean['Afwd'].values, 'r-')
    trans = mtransforms.blended_transform_factory(ax5.transData, ax5.transAxes)
    ax5.fill_between(df2mean['align'].values, (df2mean['Afwd'] + df2std['Afwd']).values, (df2mean['Afwd'] - df2std['Afwd']).values, color='r', alpha=0.1, zorder=2)
    plt.axhline(y=0, color='k')
    ax5.fill_between(df2mean['align'].values, 0, 1, where=df2mean['laser_state'].values>0.9, facecolor='Yellow', alpha=0.3, transform=trans)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Fwd Acceleration (pixels/s^2) +/- STD')
    ax5.set_ylim([-1000,1000])


    #rotation overlay:
    ax6 = fig2.add_subplot(3,1,3)
    ax6.plot(df2mean['align'].values, df2mean['dorientation'].values, 'k-')
    trans = mtransforms.blended_transform_factory(ax6.transData, ax6.transAxes)
    ax6.fill_between(df2mean['align'].values, (df2mean['dorientation'] + df2std['dorientation']).values, (df2mean['dorientation'] - df2std['dorientation']).values, color='r', alpha=0.1, zorder=2)
    ax6.set_xlabel('Time (s)')
    ax6.fill_between(df2mean['align'].values, 0, 1, where=df2mean['laser_state'].values>0.9, facecolor='Yellow', alpha=0.3, transform=trans)
    ax6.set_ylabel('angular rotation (radians/s) +/- STD')
    ax6.set_ylim([-20,30])
    plt.subplots_adjust(bottom=0.06, top=0.98, hspace=0.31)
    """
    plt.savefig((path + "/Velocity_calculations/overlaid_plot.png"))
    plt.savefig((path + "/Velocity_calculations/overlaid_plot.svg"))


if __name__ == "__main__":
    CTRL_GENOTYPE = 'ctrl' #black
    EXP_GENOTYPE = 'MW' #blue

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
