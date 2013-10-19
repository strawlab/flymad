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

CTRL_GENOTYPE = 'ctrl' #black
EXP_GENOTYPE = 'MW' #blue

if len(sys.argv) !=2:
    print 'call flymad_velocity with directory. example: "home/user/foo/filedir"'
    exit()	

pooldf = DataFrame()
df2 = DataFrame()
if not os.path.exists(sys.argv[1] + "/Velocity_calculations/"):
    os.makedirs(sys.argv[1] + "/Velocity_calculations/")
for csvfile in sorted(glob.glob(sys.argv[1] + "/*.csv")):
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
    
#    df['smoothv'] = np.nan
#    df['smooththeta'] = np.nan
#    df['smoothAfwd'] = np.nan
#    df['smoothrot'] = np.nan
#    for row in df.index:
#        if row >=2:
#          df.ix[row,'smoothv'] = df.ix[(row-2):(row+2), 'Vfwd'].mean()
#          df.ix[row,'smooththeta'] = df.ix[(row-2):(row+2), 'orientation'].mean()
#          dt = df.ix[row, 'tracked_t'] - df.ix[row-1, 'tracked_t']
#          df.ix[row,'smoothAfwd'] = (df.ix[row, 'smoothv'] - df.ix[row-1, 'smoothv']) / dt
#          df.ix[row, 'smoothrot'] = abs(df.ix[row, 'smooththeta'] - df.ix[row-1, 'smooththeta']) / dt 
#        else:
#          continue 
#    df['smoothAfwd'][abs(df['smoothAfwd']) >=2000] = np.nan
#    df['smoothrot'][abs(df['smoothrot']) >=2000] = np.nan
    
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

#    df.to_csv((sys.argv[1] + "/Velocity_calculations/" + experimentID + "_" + date + ".csv"), index=False)
print "DF2:", df2.columns
"""
means = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False)[['Vfwd', 'Afwd', 'dorientation']].mean()
stds = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False)[['Vfwd', 'Afwd', 'dorientation']].std()
means.to_csv((sys.argv[1] + "/Velocity_calculations/means.csv"))
"""
df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].mean()
df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False)[['Vfwd', 'Afwd', 'dorientation', 'laser_state']].std()
df2mean.to_csv((sys.argv[1] + "/Velocity_calculations/overlaid_means.csv"))
df2std.to_csv((sys.argv[1] + "/Velocity_calculations/overlaid_std.csv"))

#matplotlib seems sensitive to non-float colums, so convert to
#float anything we plot
for _df in [df2mean, df2std]:
    real_cols = [col for col in _df.columns if col not in ("Genotype", "lasergroup")]
    _df[ real_cols ] = _df[ real_cols ].astype(float)

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


plt.savefig((sys.argv[1] + "/Velocity_calculations/singletrace_plot.png"))
"""
#PLOT OVERLAY


# half-assed, uninformed danno's tired method of grouping for plots:
MWmean = df2mean[df2mean['Genotype'] == EXP_GENOTYPE]
ctrlmean = df2mean[df2mean['Genotype']== CTRL_GENOTYPE]
MWstd = df2std[df2std['Genotype'] == EXP_GENOTYPE]
ctrlstd = df2std[df2std['Genotype']== CTRL_GENOTYPE]


fig2 = plt.figure()
#velocity overlay:
ax4 = fig2.add_subplot(1,1,1)
ax4.plot(MWmean['align'].values, MWmean['Vfwd'].values, 'b-', zorder=10, lw=3)
trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.fill_between(MWmean['align'].values, (MWmean['Vfwd'] + MWstd['Vfwd']).values, (MWmean['Vfwd'] - MWstd['Vfwd']).values, color='b', alpha=0.1, zorder=5)
plt.axhline(y=0, color='k')
ax4.fill_between(MWmean['align'].values, 0, 1, where=MWmean['laser_state'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
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
plt.savefig((sys.argv[1] + "/Velocity_calculations/overlaid_plot.png"))
plt.savefig((sys.argv[1] + "/Velocity_calculations/overlaid_plot.svg"))


plt.show()
