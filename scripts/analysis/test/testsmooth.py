import madplot

import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

BAG = '/mnt/strawscience/data/FlyMAD/MW/14_11/Moonw_b_5_90/2013-11-14-01-43-31.bag'
oid = 18913

arena = madplot.Arena('mm')

def get_df(smooth):

    smoothstr = {True:'smooth',False:'nosmooth'}[smooth]

    geom, dfs = madplot.load_bagfile(BAG, arena, smooth=smooth)
    t_df = dfs["tracked"]

    fig = plt.figure('traj %s' % smoothstr)
    fig.suptitle('trajectory %s' % smoothstr)
    ax = fig.add_subplot(1,1,1)
    madplot.plot_tracked_trajectory(ax, t_df, arena, debug_plot=False)
    ax.add_patch(arena.get_patch(fill=False))
    ax.set_xlabel('position (%s)' % arena.unit)
    fig.savefig('testsmooth_mwb_pos_%s.png' % smoothstr)

    df = t_df[t_df['tobj_id'] == oid]

    #print df.iloc[39].index[0] - df.iloc[0]

    times = [(ix - df.index[0]).total_seconds() for ix in df.index]

    #print times[39]

    ######yyyyyyyyyy
    fig = plt.figure('y %s' % smoothstr,figsize=(10,8))
    fig.suptitle('y %s' % smoothstr)
    ax = fig.add_subplot(2,2,1)
    ax.set_title('position')
    ax.plot(times, df['y_px'].values)
    ax.set_ylabel('y (px/s)')
    ax = fig.add_subplot(2,2,2)
    ax.set_title('converted position')
    ax.plot(times, df['y'].values)
    ax.set_ylabel('y (%s)' % arena.unit)
    ax = fig.add_subplot(2,2,3)
    ax.set_title('velocity')
    ax.plot(times, df['vy_px'].values)
    ax.set_ylabel('y-velocity (px/s)')
    ax = fig.add_subplot(2,2,4)
    ax.set_title('converted velocity')
    ax.plot(times, df['vy'].values)
    ax.set_ylabel('y-velocity (%s/s)' % arena.unit)
    fig.savefig('testsmooth_mwb_y_%s.png' % smoothstr)

    ######xxxxxxxxxxx
    fig = plt.figure('x %s' % smoothstr,figsize=(10,8))
    fig.suptitle('x %s' % smoothstr)
    ax = fig.add_subplot(2,2,1)
    ax.set_title('position')
    ax.plot(times, df['x_px'].values)
    ax.set_ylabel('x (px/s)')
    ax = fig.add_subplot(2,2,2)
    ax.set_title('converted position')
    ax.plot(times, df['x'].values)
    ax.set_ylabel('x (%s)' % arena.unit)
    ax = fig.add_subplot(2,2,3)
    ax.set_title('velocity')
    ax.plot(times, df['vx_px'].values)
    ax.set_ylabel('x-velocity (px/s)')
    ax = fig.add_subplot(2,2,4)
    ax.set_title('converted velocity')
    ax.plot(times, df['vx'].values)
    ax.set_ylabel('x-velocity (%s/s)' % arena.unit)
    fig.savefig('testsmooth_mwb_x_%s.png' % smoothstr)

    #######
    df = t_df[t_df['tobj_id'] == oid].resample('500L')
    times = [(ix - df.index[0]).total_seconds() for ix in df.index]

    fig = plt.figure('speed %s' % smoothstr,figsize=(8,4))
    fig.suptitle('speed %s' % smoothstr)
    ax = fig.add_subplot(1,2,1)
    ax.plot(times, df['v_px'].values)
    ax.set_ylabel('speed (px/s)')
    ax = fig.add_subplot(1,2,2)
    ax.plot(times, df['v'].values)
    ax.set_ylabel('speed (%s/s)' % arena.unit)
    fig.savefig('testsmooth_mwb_speed_%s.png' % smoothstr)

    return t_df[t_df['tobj_id'] == oid]

if __name__ == "__main__":

    smoothed = get_df(True)
    unsmoothed = get_df(False)

    fig = plt.figure('both')
    ax = fig.add_subplot(1,1,1)
    ax.set_title("comparing smoothed and not")

    ax.set_aspect('equal')
    xlim,ylim = arena.get_limits()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ds = 1
    ax.plot(smoothed['x'].values[::ds], smoothed['y'].values[::ds], 'r.')
    ax.plot(unsmoothed['x'].values[::ds], unsmoothed['y'].values[::ds], 'b.')
    fig.savefig('testsmooth_mwb_compare.png')

    plt.show()




