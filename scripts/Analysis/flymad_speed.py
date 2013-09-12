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
    df = pd.read_csv(csvfile)
    df['laser_state'] = df['laser_state'].fillna(value=0)
    lasermask = df[df['laser_state'] == 1]
    df = df.iloc[range(0,len(df),2)] #delete duplicated values... silly bug.
    df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)
    df = df.drop('Unnamed: 0',1)

    #MAXIMUM SPEED = 300:
    flyingmask = df['v'] >= 300
    df.ix[flyingmask, 'v'] = np.nan    
  
    #bin to 10 millisecond bins:
    df = df[np.isfinite(df['tracked_t'])]
    df['tracked_t'] = df['tracked_t'] * 100 
    df['tracked_t'] = df['tracked_t'].astype(int)
    df['tracked_t'] = df['tracked_t'].astype(float)
    df['tracked_t'] = df['tracked_t'] / 100
    df = df.groupby(df['tracked_t'], axis=0).mean() 
    df['r'] = range(0,len(df))
    df = df.set_index('r') 
    df['laser_state'][df['laser_state'] > 0] = 1
    dfshift = df.shift()    

    df['Genotype'] = genotype
    df['lasergroup'] = laser
    df['RepID'] = repID

    #combine 7s trials together into df2:
    laserons = df[(df['laser_state']-dfshift['laser_state']) == 1] 
    for row in laserons.index:
        dftemp = df.ix[(row-990):(row+5800),:]
        print dftemp.shape
        print len(np.arange(-9.90,58.01,0.01))
        dftemp['align'] = np.arange(-9.90,58.01,0.01)
        df2 = pd.concat([df2, dftemp])

    pooldf = pd.concat([pooldf, df])   

    df.to_csv((sys.argv[1] + "/Speed_calculations/" + experimentID + "_" + date + ".csv"), index=False)


pooldfmean = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False).mean()
pooldfstd = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t']).std()
pooldfn = pooldf.groupby(['Genotype', 'lasergroup','tracked_t']).count()
pooldfmean.sort(['tracked_t'])

df = pooldfmean
df.to_csv((sys.argv[1] + "/Speed_calculations/means.csv"))


df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).mean()
df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).std()
df2mean.to_csv((sys.argv[1] + "/Speed_calculations/overlaid_means.csv"))
df2std.to_csv((sys.argv[1] + "/Speed_calculations/overlaid_std.csv"))

# PLOT FULL TRACE (MEAN +- STD)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(df['tracked_t'], df['v'], 'b-', zorder=1)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.fill_between((df['tracked_t']).values, 0, 1, where=(df['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (pixels/s)')
ax.set_xlim([-10,180])
plt.axhline(y=0, color='k')

plt.savefig((sys.argv[1] + "/Speed_calculations/singletrace_plot.png"))

#PLOT OVERLAY (MEAN +- STD)

fig2 = plt.figure()
ax4 = fig2.add_subplot(1,1,1)
ax4.plot(df2mean['align'], df2mean['v'], 'b-')
trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.fill_between((df2mean['align']).values, (df2mean['v'] + df2std['v']).values, (df2mean['v'] - df2std['v']).values, color='b', alpha=0.1, zorder=2)
plt.axhline(y=0, color='k')
ax4.fill_between((df2mean['align']).values, 0, 1, where=(df2mean['laser_power']>0).values, facecolor='Yellow', alpha=0.1, transform=trans)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Speed (pixels/s)')

plt.subplots_adjust(bottom=0.06, top=0.98, hspace=0.31)

plt.savefig((sys.argv[1] + "/Speed_calculations/overlaid_plot.png"))


plt.show()
