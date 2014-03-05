import sys
import json
import os.path
import matplotlib.pyplot as plt

import madplot

def plot_trajectory(ax,arena,ldf,tdf,hdf,geom):
    madplot.plot_tracked_trajectory(ax, tdf, arena,
                debug_plot=True,
                color='k',
    )
    ax.add_patch(arena.get_patch(color='k', alpha=0.1))

    patch = arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2)
    if patch is not None:
        ax.add_patch(patch)

def plot_laser(ax,arena,ldf,tdf,hdf,geom):
    madplot.plot_laser_trajectory(ax, ldf, arena)
    ax.add_patch(arena.get_patch(color='k', alpha=0.1))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag file')
    parser.add_argument('--unit', default=False)
    parser.add_argument('--no-smooth', dest='smooth', action='store_false', default=True)

    args = parser.parse_args()
    path = args.path[0]

    arena = madplot.Arena(args.unit)
    geom, dfs = madplot.load_bagfile(path, arena, smooth=args.smooth)
    ldf = dfs["targeted"]
    tdf = dfs["tracked"]
    hdf = dfs["ttm"]

    axt = plt.figure("Trajectory").add_subplot(1,1,1)
    plot_trajectory(axt,arena,ldf,tdf,hdf,geom)

    axl = plt.figure("Laser").add_subplot(1,1,1)
    plot_laser(axl,arena,ldf,tdf,hdf,geom)

    for ax in (axt,axl):
        ax.set_xlabel('x (%s)' % arena.unit)
        ax.set_ylabel('y (%s)' % arena.unit)

    axdt = plt.figure("Tracking").add_subplot(1,1,1)
    axvel = plt.figure("Velocity").add_subplot(1,1,1)
    for obj_id,df in tdf.groupby('tobj_id'):
        axdt.plot(df['t_dt'].values, ',')
        t = df['t_ts'].values
        axvel.plot(t - t[0], df['v'].values, ',', label=str(obj_id))

    axdt.set_xlabel('observation (n)')
    axdt.set_ylabel('observation dt (s)')
    axvel.legend()
    axvel.set_xlabel('t (s)')
    axvel.set_ylabel('velocity (%s/s)' % arena.unit)

    plt.show()
