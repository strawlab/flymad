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

from trackingparams import Kalman

REPLACE_XY = False

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
        experimentID,date,time = csvfilefn.split("_",2)
        genotype,laser,repID = experimentID.split("-",2)
        repID = repID + "_" + date
        print "processing: ", experimentID
    except:
        print "invalid filename:", csvfilefn
        continue 
    df = pd.read_csv(csvfile)

    if REPLACE_XY:
        nx,ny,nvx,nvy,nv = "x","y","vx","vy","v"
    else:
        nx,ny,nvx,nvy,nv = "sx","sy","svx","svy","sv"

    #smooth the positions, and recalculate the velocitys based on this. if
    #REPLACE_XY is true, the old columns (x,y,vx,vy,v) are replaced, otherwise
    #new colums with the smoothed values are added (sx,sy,svx,svy,sv).
    kf = Kalman()
    smoothed = kf.smooth(df['x'], df['y'])
    df[nx] = smoothed[:,0]
    df[ny] = smoothed[:,1]
    dt = np.gradient(df['tracked_t'].values)
    df[nvx] = np.gradient(df[nx].values) / dt
    df[nvy] = np.gradient(df[nx].values) / dt
    df[nv] = np.sqrt( (df[nvx].values**2) + (df[nvy].values**2) )

    df['laser_state'] = df['laser_state'].fillna(value=0)
    lasermask = df[df['laser_state'] == 1]
    df = df.iloc[range(0,len(df),2)] #delete duplicated values... silly bug.
    df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)
    df = df.drop('Unnamed: 0',1)   

    #ROTATE by pi if orientation is east
    df['zx'][df['zx'] == 'z'] = math.pi
    df['zx'][df['zx'] == 'x'] = 0
    df['orientation'] = df['theta'] + df['zx']
    #ROTATE by pi if orientation is north/south (plusminus 0.25pi) and hemisphere does not match scoring:
    smask = df[df['as'] == 'a']
    smask = smask[smask['orientation'] < 0.75*(math.pi)]
    smask = smask[smask['orientation'] > 0.25*(math.pi)]
    amask = df[df['as'] == 's']
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
    flyingmask = df['v'] >= 300
    df.ix[flyingmask, 'v'] = np.nan    

    #CALCULATE FORWARD VELOCITY
    df['Vtheta'] = np.arctan2(df['vy'], df['vx'])
    df['Vfwd'] = (np.cos(df['orientation'] - df['Vtheta'])) * df['v']

    
    #bin to 10 millisecond bins:
    df = df[np.isfinite(df['tracked_t'])]
    df['tracked_t'] = df['tracked_t'] * 100 
    df['tracked_t'] = df['tracked_t'].astype(int)
    df['tracked_t'] = df['tracked_t'].astype(float)
    df['tracked_t'] = df['tracked_t'] / 100
    df = df.groupby(df['tracked_t'], axis=0).mean() 
    df['r'] = range(0,len(df))
    df = df.set_index('r') 
    df['laser_state'][df['laser_state'] > 0] = 1 #silly bug from combining timestamps
        
    #CALCULATE ANGULAR VELOCITY and ACCELERATION
    dfshift = df.shift()
    df['deltaT'] = df['tracked_t'] - dfshift['tracked_t']
    df['Rotation'] = abs(df['orientation'] - dfshift['orientation']) / df['deltaT']
    df['Afwd'] = (df['Vfwd'] - dfshift['Vfwd']) / df['deltaT']
    #EXCLUDE JUMPS (too fast rotation and acceleration)
    df['Rotation'][df['Rotation'] >= 5*math.pi] = np.nan
   

    df['Genotype'] = genotype
    df['lasergroup'] = laser
    df['RepID'] = repID
    
    df['smoothv'] = np.nan
    df['smooththeta'] = np.nan
    df['smoothAfwd'] = np.nan
    df['smoothrot'] = np.nan
    for row in df.index:
        if row >=2:
          df.ix[row,'smoothv'] = df.ix[(row-2):(row+2), 'Vfwd'].mean()
          df.ix[row,'smooththeta'] = df.ix[(row-2):(row+2), 'orientation'].mean()
          dt = df.ix[row, 'tracked_t'] - df.ix[row-1, 'tracked_t']
          df.ix[row,'smoothAfwd'] = (df.ix[row, 'smoothv'] - df.ix[row-1, 'smoothv']) / dt
          df.ix[row, 'smoothrot'] = abs(df.ix[row, 'smooththeta'] - df.ix[row-1, 'smooththeta']) / dt 

        else:
          continue 
    df['smoothAfwd'][abs(df['smoothAfwd']) >=2000] = np.nan
    df['smoothrot'][abs(df['smoothrot']) >=2000] = np.nan
    
    #combine 7s trials together into df2:
    laserons = df[(df['laser_state']-dfshift['laser_state']) == 1] 
    for row in laserons.index:
        dftemp = df.ix[(row-195):(row+695),:]
        dftemp['align'] = np.arange(-1.95,6.96,0.01)
        df2 = pd.concat([df2, dftemp])

    pooldf = pd.concat([pooldf, df])   

    df.to_csv((sys.argv[1] + "/Velocity_calculations/" + experimentID + "_" + date + ".csv"), index=False)


pooldfmean = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t'], as_index=False).mean()
pooldfstd = pooldf.groupby(['Genotype', 'lasergroup', 'tracked_t']).std()
#pooldfn = pooldf.groupby(['Genotype', 'lasergroup','tracked_t']).count()
pooldfmean.sort(['tracked_t'])

df = pooldfmean
df.to_csv((sys.argv[1] + "/Velocity_calculations/means.csv"))


df2mean = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).mean()
df2std = df2.groupby(['Genotype', 'lasergroup','align'], as_index=False).std()
df2mean.to_csv((sys.argv[1] + "/Velocity_calculations/overlaid_means.csv"))
df2std.to_csv((sys.argv[1] + "/Velocity_calculations/overlaid_std.csv"))

# PLOT
print df2.shape
fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.plot(df['tracked_t'], df['smoothv'], 'b-', zorder=1)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.fill_between((df['tracked_t']).values, 0, 1, where=(df['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Fwd Velocity (pixels/s)')
ax.set_xlim([-10,70])
plt.axhline(y=0, color='k')

ax2 = fig.add_subplot(3,1,2)
ax2.plot((df['tracked_t']), (df['smoothAfwd']), 'r-')
trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
ax2.fill_between((df['tracked_t']).values, 0, 1, where=(df['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Fwd Acceleration (pixels/s^2)')
ax2.set_xlim([-10,70])
plt.axhline(y=0, color='k')

ax3 = fig.add_subplot(3,1,3)
ax3.plot((df['tracked_t']), (df['smoothrot']), 'k-')
trans = mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
ax3.fill_between((df['tracked_t']).values, 0, 1, where=(df['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Rotation (Radians/s)')
ax3.set_xlim([-10,70])


plt.savefig((sys.argv[1] + "/Velocity_calculations/singletrace_plot.png"))

#PLOT OVERLAY

fig2 = plt.figure()
#velocity overlay:
ax4 = fig2.add_subplot(3,1,1)
ax4.plot(df2mean['align'], df2mean['smoothv'], 'b-')
trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.fill_between((df2mean['align']).values, (df2mean['smoothv'] + df2std['smoothv']).values, (df2mean['smoothv'] - df2std['smoothv']).values, color='b', alpha=0.1, zorder=2)
plt.axhline(y=0, color='k')
ax4.fill_between((df2mean['align']).values, 0, 1, where=(df2mean['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Fwd Velocity (pixels/s)')

#acceleration overlay:
ax5 = fig2.add_subplot(3,1,2)
ax5.plot(df2mean['align'], df2mean['smoothAfwd'], 'r-')
trans = mtransforms.blended_transform_factory(ax5.transData, ax5.transAxes)
ax5.fill_between((df2mean['align']).values, ((df2mean['smoothAfwd']).values + (df2std['smoothAfwd']).values), (df2mean['smoothAfwd'].values - df2std['smoothAfwd'].values), color='r', alpha=0.1, zorder=2)
plt.axhline(y=0, color='k')
ax5.fill_between((df2mean['align']).values, 0, 1, where=(df2mean['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Fwd Acceleration (pixels/s^2)')


#rotation overlay:
ax6 = fig2.add_subplot(3,1,3)
ax6.plot(df2mean['align'], df2mean['smoothrot'], 'k-')
trans = mtransforms.blended_transform_factory(ax6.transData, ax6.transAxes)
ax6.fill_between((df2mean['align']).values, ((df2mean['smoothrot']).values + (df2std['smoothrot']).values), (df2mean['smoothrot'].values - df2std['smoothrot'].values), color='r', alpha=0.1, zorder=2)
ax6.set_xlabel('Time (s)')
ax6.fill_between((df2mean['align']).values, 0, 1, where=(df2mean['laser_power']>0).values, facecolor='Yellow', alpha=0.3, transform=trans)
ax6.set_ylabel('angular rotation (radians/s)')
plt.subplots_adjust(bottom=0.06, top=0.98, hspace=0.31)

plt.savefig((sys.argv[1] + "/Velocity_calculations/overlaid_plot.png"))


plt.show()
