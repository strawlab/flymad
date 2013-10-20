import json
import os.path
import datetime

import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches

import roslib
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
        return list(inter.exterior.coords)

    def get_intersect_patch(self, geom, **kwargs):
        pts = self.get_intersect_points(geom)
        return matplotlib.patches.Polygon(pts, **kwargs)

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

#        ax.plot(fly_x[0],fly_y[0],'b.')

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
        _df = group.resample('20L')
        ax.plot(_df['x'],_df['y'],'k-')


def load_bagfile(bagpath):
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

    l_df = pd.DataFrame(l_data, index=l_index)
    l_df['time'] = l_df.index.values.astype('datetime64[ns]')
    l_df.set_index(['time'], inplace=True)

    t_df = pd.DataFrame(t_data, index=t_index)
    t_df['time'] = t_df.index.values.astype('datetime64[ns]')
    t_df.set_index(['time'], inplace=True)


    points_x = [pt.x for pt in geom_msg.points]
    points_y = [pt.y for pt in geom_msg.points]

    return l_df, t_df, (points_x, points_y)

def calculate_time_in_area(tdf, arena, geom, interval=30):
    poly = arena.get_intersect_polygon(geom)

    pct = []
    offset = []

    #put the df into 10ms bins
    df = tdf.resample('10L')

    #maybe pandas has a built in way to do this?
    t0 = t1 = df.index[0]
    tend = df.index[-1]
    toffset = 0

    while t1 < tend:
        t1 = t0 + datetime.timedelta(seconds=interval)
        
        #get trajectories of that time
        idf = df[t0:t1]

        npts = 0
        for idx,ser in idf.iterrows():
            pt = sg.Point(ser['x'], ser['y'])
            #check if point is in the target area
            npts += int(poly.contains(pt))

        #percentage of time in area
        pct.append( 100.0 * (npts / float(len(idf))) )
        offset.append( toffset )

        t0 = t1
        toffset += interval

    return offset, pct

