import madplot

import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

BAG = '/mnt/strawscience/data/FlyMAD/velocity_test/2013-11-21-18-00-00.bag'
oid = 52251

smooth = True
smoothstr = {True:'smooth',False:'nosmooth'}[smooth]

arena = madplot.Arena('mm')
geom, dfs = madplot.load_bagfile(BAG, arena, smooth=smooth)
l_df = dfs["targeted"]
t_df = dfs["tracked"]
h_df = dfs["ttm"]

fig = plt.figure('traj %s' % smoothstr)
fig.suptitle('trajectory %s' % smoothstr)
ax = fig.add_subplot(1,1,1)
madplot.plot_tracked_trajectory(ax, t_df, arena, debug_plot=False)
ax.add_patch(arena.get_patch(fill=False))
ax.set_xlabel('position (%s)' % arena.unit)
fig.savefig('dorothea_pos_%s.png' % smoothstr)

df = t_df[t_df['tobj_id'] == oid].resample('100L')

#print df.iloc[39].index[0] - df.iloc[0]

times = [(ix - df.index[0]).total_seconds() for ix in df.index]

#print times[39]

######yyyyyyyyyy
fig = plt.figure('y %s' % smoothstr,figsize=(10,8))
fig.suptitle('y %s' % smoothstr)
ax = fig.add_subplot(2,2,1)
ax.set_title('position')
ax.plot(times, df['y_px'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('y (px/s)')
ax = fig.add_subplot(2,2,2)
ax.set_title('converted position')
ax.plot(times, df['y'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('y (%s)' % arena.unit)
ax = fig.add_subplot(2,2,3)
ax.set_title('velocity')
ax.plot(times, df['vy_px'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('y-velocity (px/s)')
ax = fig.add_subplot(2,2,4)
ax.set_title('converted velocity')
ax.plot(times, df['vy'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('y-velocity (%s/s)' % arena.unit)
fig.savefig('dorothea_y_%s.png' % smoothstr)

######xxxxxxxxxxx
fig = plt.figure('x %s' % smoothstr,figsize=(10,8))
fig.suptitle('x %s' % smoothstr)
ax = fig.add_subplot(2,2,1)
ax.set_title('position')
ax.plot(times, df['x_px'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('x (px/s)')
ax = fig.add_subplot(2,2,2)
ax.set_title('converted position')
ax.plot(times, df['x'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('x (%s)' % arena.unit)
ax = fig.add_subplot(2,2,3)
ax.set_title('velocity')
ax.plot(times, df['vx_px'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('x-velocity (px/s)')
ax = fig.add_subplot(2,2,4)
ax.set_title('converted velocity')
ax.plot(times, df['vx'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('x-velocity (%s/s)' % arena.unit)
fig.savefig('dorothea_x_%s.png' % smoothstr)

#######
df = t_df[t_df['tobj_id'] == oid].resample('500L')
times = [(ix - df.index[0]).total_seconds() for ix in df.index]

fig = plt.figure('speed %s' % smoothstr,figsize=(8,4))
fig.suptitle('speed %s' % smoothstr)
ax = fig.add_subplot(1,2,1)
ax.plot(times, df['v_px'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('speed (px/s)')
ax = fig.add_subplot(1,2,2)
ax.plot(times, df['v'].values)
ax.axvline(3.9,color='k');ax.axvline(7.5,color='k');
ax.set_ylabel('speed (%s/s)' % arena.unit)
fig.savefig('dorothea_speed_%s.png' % smoothstr)


plt.show()




