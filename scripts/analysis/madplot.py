import json
import os.path
import datetime
import collections

import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches

import roslib; roslib.load_manifest('rosbag')
import rosbag

class Arena:
    def __init__(self, jsonconf=dict()):
        x = jsonconf.get('cx',360)
        y = jsonconf.get('cy',255)
        r = jsonconf.get('cr',200)
        self._x, self._y, self._r = x,y,r
        self._circ = sg.Point(x,y).buffer(r)

    def get_intersect_polygon(self, geom):
        poly = sg.Polygon(list(zip(*geom)))
        inter = self._circ.intersection(poly)
        return inter

    def get_intersect_points(self, geom):
        inter = self.get_intersect_polygon(geom)
        if inter:
            return list(inter.exterior.coords)
        else:
            return []

    def get_intersect_patch(self, geom, **kwargs):
        pts = self.get_intersect_points(geom)
        if pts:
            return matplotlib.patches.Polygon(pts, **kwargs)
        return None

    def get_patch(self, **kwargs):
        return matplotlib.patches.Circle((self._x,self._y), radius=self._r, **kwargs)

    def get_limits(self):
        #(xlim, ylim)
        return (150,570), (47,463)

def plot_geom(ax, geom):
    ax.plot(geom[0],geom[1],'g-')

def plot_laser_trajectory(ax, df, plot_starts=False, plot_laser=False, intersect_patch=None, limits=None):
    ax.set_aspect('equal')

    if limits is not None:
        xlim,ylim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if intersect_patch is not None:
        ax.add_patch(intersect_patch)

    first = True
    for name, group in df.groupby('obj_id'):
        fly_x = group['fly_x'].values
        fly_y = group['fly_y'].values

        pp = ax.plot(fly_x,fly_y,'k.',label="predicted" if first else "__nolegend__")

#        ax.plot(fly_x[0],fly_y[0],'b.',label="predicted" if first else "__nolegend__")

        #plot the laser when under fine control
        laserdf = group[group['mode'] == 2]
        lp = ax.plot(laserdf['laser_x'],laserdf['laser_y'],'r.',label="required" if first else "__nolegend__")

        first = False

def plot_tracked_trajectory(ax, df, intersect_patch=None, limits=None):
    ax.set_aspect('equal')

    if limits is not None:
        xlim,ylim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if intersect_patch is not None:
        ax.add_patch(intersect_patch)

    for name, group in df.groupby('obj_id'):
        print "\t%s:%s" % (name,len(group))
        _df = group.resample('20L')
        ax.plot(_df['x'],_df['y'],'k-')

def load_bagfile(bagpath, arena):
    def in_area(row, poly):
        pt = sg.Point(row['x'], row['y'])
        return pd.Series({"in_area":poly.contains(pt)})

    print "loading", bagpath
    bag = rosbag.Bag(bagpath)

    l_index = []
    l_data = {k:[] for k in ("obj_id","fly_x","fly_y","laser_x","laser_y","laser_power","mode")}
    l_data_names = l_data.keys()

    t_index = []
    t_data = {k:[] for k in ("obj_id","x","y")}
    t_data_names = t_data.keys()

    geom_msg = None

    for topic,msg,rostime in bag.read_messages(topics=["/targeter/targeted",
                                                       "/flymad/tracked",
                                                       "/draw_geom/poly"]):
        if topic == "/draw_geom/poly":
            geom_msg = msg
        elif topic == "/targeter/targeted":
            l_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
            for k in l_data_names:
                l_data[k].append( getattr(msg,k) )
        elif topic == "/flymad/tracked":
            if msg.is_living:
                t_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
                t_data['obj_id'].append(msg.obj_id)
                t_data['x'].append(msg.state_vec[0])
                t_data['y'].append(msg.state_vec[1])

    points_x = [pt.x for pt in geom_msg.points]
    points_y = [pt.y for pt in geom_msg.points]
    geom = (points_x, points_y)

    poly = arena.get_intersect_polygon(geom)

    l_df = pd.DataFrame(l_data, index=l_index)
    l_df['time'] = l_df.index.values.astype('datetime64[ns]')
    l_df.set_index(['time'], inplace=True)

    t_df = pd.DataFrame(t_data, index=t_index)
    t_df['time'] = t_df.index.values.astype('datetime64[ns]')
    t_df.set_index(['time'], inplace=True)

    #add a new colum if they were in the area
    t_df = pd.concat([t_df,
                      t_df.apply(in_area, axis=1, args=(poly,))],
                      axis=1)

    #remove short trials here
    short_tracks = []
    for name, group in t_df.groupby('obj_id'):
        if len(group) < 100:
            short_tracks.append(name)

    l_df = l_df[~l_df['obj_id'].isin(short_tracks)]
    t_df = t_df[~t_df['obj_id'].isin(short_tracks)]

    return l_df, t_df, geom

def calculate_time_in_area(df, maxtime=None, interval=20):
    pct = []
    offset = []

    #put the df into 10ms bins
    #df = tdf.resample('10L')

    #maybe pandas has a built in way to do this?
    t0 = t1 = df.index[0]
    tend = df.index[-1] if maxtime else (t0 + datetime.timedelta(seconds=maxtime))
    toffset = 0

    while t1 < tend:
        if maxtime:
            t1 = t0 + datetime.timedelta(seconds=interval)
        else:
            t1 = min(t0 + datetime.timedelta(seconds=interval), tend)
        
        #get trajectories of that timespan
        idf = df[t0:t1]
        #number of timepoints in the area
        npts = idf['in_area'].sum()
        #percentage of time in area
        pct.append( 100.0 * (npts / float(len(idf))) )

        offset.append( toffset )

        t0 = t1
        toffset += interval

    return offset, pct

def calculate_time_to_area(tdf, maxtime=300):
    tta = []

    for name, group in tdf.groupby('obj_id'):
        t0 = group.head(1)
        t0_in_area = t0['in_area']
        t1_in_area = False
        for ix,row in group.iterrows():
            if row['in_area']:
                t1_in_area = True
                break

        #did the fly finish inside, and start outside
        if t1_in_area and (not t0_in_area):
            dt = ix - t0.index[0]
            tta.append( dt )

    print tta



