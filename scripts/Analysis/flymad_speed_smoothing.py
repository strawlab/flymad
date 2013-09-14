import math
import glob
import os
import sys
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import flymad_analysis

if len(sys.argv) !=2:
    print 'call flymad_speed with directory. example: "home/user/foo/filedir"'
    exit()	

pooldf = DataFrame()
df2 = DataFrame()
if not os.path.exists(sys.argv[1] + "/Speed_calculations/"):
    os.makedirs(sys.argv[1] + "/Speed_calculations/")
for csvfile in sorted(glob.glob(sys.argv[1] + "/*.csv")):
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

    #pandas doesn't know how to resample columns with type 'object', and 
    #this 
    flymad_analysis.fix_scoring_colums(df,
                        valmap={'zx':{'z':1,'x':0},
                                'as':{'a':1,'s':0}})

    #resample to 10ms (mean) and set a proper time index on the df
    df = flymad_analysis.fixup_index_and_resample(df, '10L')

    #smooth the positions, and recalculate the velocitys based on this.
    dt = flymad_analysis.kalman_smooth_dataframe(df)

    df['laser_state'] = df['laser_state'].fillna(value=0)
    lasermask = df[df['laser_state'] == 1]

    df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)

    #MAXIMUM SPEED = 300:
    df[df['v'] >= 30]['v'] = np.nan

    df = df.groupby(df['tracked_t'], axis=0).mean() 
    df['laser_state'][df['laser_state'] > 0] = 1

    df['Genotype'] = genotype
    df['lasergroup'] = laser
    df['RepID'] = repID
    """
    #combine 7s trials together into df2:
    laserons = df[(df['laser_state']-dfshift['laser_state']) == 1] 
    for row in laserons.index:
        #dftemp = df.ix[(row-990):(row+5800),:]
        #print dftemp.shape
        #print len(np.arange(-9.90,58.01,0.01))
        #dftemp['align'] = np.arange(-9.90,58.01,0.01)
        #df2 = pd.concat([df2, dftemp])
        before =row -10
        after = row 59
        dftemp = df.ix[before:after]
        dftemp['align'] = np.linspace(0,(after-before),len(dftemp))
        df2 = pd.concat([df2, dftemp])
    """      
    pooldf = pd.concat([pooldf, df])   

    df.to_csv((sys.argv[1] + "/Speed_calculations/" + experimentID + "_" + date + ".csv"), index=False)


df = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False).mean()
dfstd = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t']).std()
dfn = pooldf.groupby(['Genotype', 'lasergroup','tracked_t']).count()

df.to_csv((sys.argv[1] + "/Speed_calculations/means.csv"))

"""
df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).mean()
df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).std()
df2mean.to_csv((sys.argv[1] + "/Speed_calculations/overlaid_means.csv"))
df2std.to_csv((sys.argv[1] + "/Speed_calculations/overlaid_std.csv"))
"""
# PLOT FULL TRACE (MEAN +- STD)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(df['tracked_t'].values, df['v'].values, 'b-', zorder=1)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.fill_between((df['tracked_t']).values, 0, 1, where=(df['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
#ax.fill_between((df['tracked_t']).values, (df['v'] + dfstd['v']).values, (df['v'] - dfstd['v']).values, color='b', alpha=0.1, zorder=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (pixels/s)')
ax.set_xlim([-10,59])
ax.set_ylim([0,75])
#plt.axhline(y=0, color='k')
"""

#PLOT OVERLAY (MEAN +- STD)

fig2 = plt.figure()
ax4 = fig2.add_subplot(1,1,1)
ax4.plot(df2mean['align'], df2mean['v'], 'b-')
trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.fill_between((df2mean['align']).values, (df2mean['v'] + df2std['v']).values, (df2mean['v'] - df2std['v']).values, color='b', alpha=0.1, zorder=2)
plt.axhline(y=0, color='k')
ax4.fill_between((df2mean['align']).values, 0, 1, where=(df2mean['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Speed (pixels/s)')
"""
plt.subplots_adjust(bottom=0.06, top=0.98, hspace=0.31)

plt.savefig((sys.argv[1] + "/Speed_calculations/overlaid_plot.png"))


plt.show()
