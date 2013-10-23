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

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.__version__ == "0.11.0"

def prepare_data(path, exp_genotype, ctrl_genotype):

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

    expdf = df2[df2['Genotype'] == exp_genotype]
    ctrldf = df2[df2['Genotype']== ctrl_genotype]

    #we no longer need to group by genotype, and lasergroup is always the same here
    #so just drop it. 
    assert len(expdf['lasergroup'].unique()) == 1, "only one lasergroup handled"

    #Also ensure things are floats before plotting can fail, which it does because
    #groupby does not retain types on grouped colums, which seems like a bug to me

    expmean = expdf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].mean().astype(float)
    ctrlmean = ctrldf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].mean().astype(float)

    expstd = expdf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].std().astype(float)
    ctrlstd = ctrldf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].std().astype(float)

    expn = expdf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].count().astype(float)
    ctrln = ctrldf.groupby(['align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].count().astype(float)

    ####AAAAAAAARRRRRRRRRRRGGGGGGGGGGGGGGHHHHHHHHHH so much copy paste here
    df2.save(path + "/df2.df")
    expmean.save(path + "/expmean.df")
    ctrlmean.save(path + "/ctrlmean.df")
    expstd.save(path + "/expstd.df")
    ctrlstd.save(path + "/ctrlstd.df")
    expn.save(path + "/expn.df")
    ctrln.save(path + "/ctrln.df")

    return expmean, ctrlmean, expstd, ctrlstd, expn, ctrln

def load_data( path ):
    return (
            pd.load(path + "/expmean.df"),
            pd.load(path + "/ctrlmean.df"),
            pd.load(path + "/expstd.df"),
            pd.load(path + "/ctrlstd.df"),
            pd.load(path + "/expn.df"),
            pd.load(path + "/ctrln.df"),
    )


def plot_data( path, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln ):

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

    fig2 = plt.figure("Velocity Multiplot")
    ax = fig2.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    exp=dict(xaxis=expmean['align'].values,
                             value=expmean['Vfwd'].values,
                             std=expstd['Vfwd'].values,
                             n=expn['Vfwd'].values,
                             label='Moonwalker>TRPA1',
                             ontop=True),
                    ctrl=dict(xaxis=ctrlmean['align'].values,
                              value=ctrlmean['Vfwd'].values,
                              std=ctrlstd['Vfwd'].values,
                              n=ctrln['Vfwd'].values,
                              label='Control'),
                    targetbetween=ctrlmean['laser_state'].values>0,
                    downsample=2,
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fwd Velocity (pixels/s) +/- STD')
    ax.set_ylim([-60, 120])
    ax.set_xlim([0, 9])

    fig2.savefig((path + "/Velocity_calculations/vfwd_plot.png"))
    fig2.savefig((path + "/Velocity_calculations/vfwd_plot.svg"))

    if 0:
        fig3 = plt.figure("Acceleration Multiplot")
        ax = fig3.add_subplot(1,1,1)

        flymad_plot.plot_timeseries_with_activation(ax,
                        exp=dict(xaxis=expmean['align'].values,
                                 value=expmean['Afwd'].values,
                                 std=expstd['Afwd'].values,
                                 n=expn['Afwd'].values,
                                 ontop=True),
                        ctrl=dict(xaxis=ctrlmean['align'].values,
                                  value=ctrlmean['Afwd'].values,
                                  std=ctrlstd['Afwd'].values,
                                  n=ctrln['Afwd'].values),
                        targetbetween=ctrlmean['laser_state'].values>0
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fwd Acceleration (pixels/s^2) +/- STD')
        ax.set_ylim([-1000, 1000])
        ax.set_xlim([0, 9])

        fig3.savefig((path + "/Velocity_calculations/afwd_plot.png"))
        fig3.savefig((path + "/Velocity_calculations/afwd_plot.svg"))

    if 0:
        fig4 = plt.figure("Angular Rotation Multiplot")
        ax = fig4.add_subplot(1,1,1)

        flymad_plot.plot_timeseries_with_activation(ax,
                        exp=dict(xaxis=expmean['align'].values,
                                 value=expmean['dorientation'].values,
                                 std=expstd['dorientation'].values,
                                 n=expn['dorientation'].values,
                                 ontop=True),
                        ctrl=dict(xaxis=ctrlmean['align'].values,
                                  value=ctrlmean['dorientation'].values,
                                  std=ctrlstd['dorientation'].values,
                                  n=ctrln['dorientation'].values),
                        targetbetween=ctrlmean['laser_state'].values>0
        )
        ax.set_xlabel('Time (s)')
        ax.set_xlim([0, 9])
        ax.set_ylabel('angular rotation (radians/s) +/- STD')
        ax.set_ylim([-20,30])

        fig4.savefig((path + "/Velocity_calculations/dorientation_plot.png"))
        fig4.savefig((path + "/Velocity_calculations/dorientation_plot.svg"))


if __name__ == "__main__":
    CTRL_GENOTYPE = 'ctrl' #black
    EXP_GENOTYPE = 'MW' #blue

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

    plot_data(path, *data)

    if args.show:
        plt.show()

