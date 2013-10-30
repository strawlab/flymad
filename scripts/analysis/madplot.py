import json
import os.path
import datetime
import math
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
        if geom:
            poly = sg.Polygon(list(zip(*geom)))
            inter = self._circ.intersection(poly)
            return inter
        else:
            return None

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

def get_path(path, dat, bname):
    bpath = dat.get('_base',os.path.abspath(os.path.dirname(path)))
    return os.path.join(bpath, bname)

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

def plot_tracked_trajectory(ax, df, limits=None, ds=1, minlen=10000, **kwargs):
    ax.set_aspect('equal')

    if limits is not None:
        xlim,ylim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for name, group in df.groupby('obj_id'):
        if len(group) < minlen:
            print "\ttraj: skipping obj_id", name, "too short", len(group)
            continue
        ax.plot(group['x'].values[::ds],group['y'].values[::ds],**kwargs)

def load_bagfile(bagpath, arena, filter_short=100):
    def in_area(row, poly):
        if poly:
            in_area = poly.contains( sg.Point(row['x'], row['y']) )
        else:
            in_area = False
        return pd.Series({"in_area":in_area})

    print "loading", bagpath
    bag = rosbag.Bag(bagpath)

    geom_msg = None

    l_index = []
    l_data = {k:[] for k in ("obj_id","fly_x","fly_y","laser_x","laser_y","laser_power","mode")}
    l_data_names = l_data.keys()

    t_index = []
    t_data = {k:[] for k in ("obj_id","x","y","vx","vy",'v','t_framenumber')}

    h_index = []
    h_data = {k:[] for k in ("head_x", "head_y", "body_x", "body_y", "target_x", "target_y", "target_type", "h_framenumber", "h_processing_time")}
    h_data_names = ("head_x", "head_y", "body_x", "body_y", "target_x", "target_y", "target_type")

    for topic,msg,rostime in bag.read_messages(topics=["/targeter/targeted",
                                                       "/flymad/tracked",
                                                       "/draw_geom/poly",
                                                       "/flymad/laser_head_delta"]):
        if topic == "/targeter/targeted":
            l_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
            for k in l_data_names:
                l_data[k].append( getattr(msg,k) )
        elif topic == "/flymad/tracked":
            if msg.is_living:
                vx = msg.state_vec[2]
                vy = msg.state_vec[3]
                t_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
                t_data['obj_id'].append(msg.obj_id)
                t_data['t_framenumber'].append(msg.framenumber)
                t_data['x'].append(msg.state_vec[0])
                t_data['y'].append(msg.state_vec[1])
                t_data['vx'].append(vx)
                t_data['vy'].append(vy)
                t_data['v'].append(math.sqrt( (vx**2) + (vy**2) ))
        elif topic == "/draw_geom/poly":
            if geom_msg is not None:
                print "WARNING: DUPLICATE GEOM MSG", msg, "vs", geom_msg
            geom_msg = msg
        elif topic == "/flymad/laser_head_delta":
            h_index.append( datetime.datetime.fromtimestamp(rostime.to_sec()) )
            for k in h_data_names:
                h_data[k].append( getattr(msg,k) )
            h_data["h_framenumber"].append( msg.framenumber )
            h_data["h_processing_time"].append( msg.processing_time )

    if geom_msg is not None:
        points_x = [pt.x for pt in geom_msg.points]
        points_y = [pt.y for pt in geom_msg.points]
        geom = (points_x, points_y)
    else:
        geom = tuple()

    poly = arena.get_intersect_polygon(geom)

    l_df = pd.DataFrame(l_data, index=l_index)
    t_df = pd.DataFrame(t_data, index=t_index)
    h_df = pd.DataFrame(h_data, index=h_index)

    #add a new colum if they were in the area
    t_df = pd.concat([t_df,
                      t_df.apply(in_area, axis=1, args=(poly,))],
                      axis=1)

    if filter_short:
        #find short trials here
        short_tracks = []
        for name, group in t_df.groupby('obj_id'):
            if len(group) < filter_short:
                print '\tload ignoring trajectory with obj_id %s (%s samples long)' % (name, len(group))
                short_tracks.append(name)

        l_df = l_df[~l_df['obj_id'].isin(short_tracks)]
        t_df = t_df[~t_df['obj_id'].isin(short_tracks)]

    return l_df, t_df, h_df, geom

def load_bagfile_single_dataframe(*args, **kwargs):
    l_df, t_df, h_df, geom = load_bagfile(*args, **kwargs)

    #merge the dataframes
    #check we have about the same amount of data
    size_similarity = min(len(t_df),len(l_df),len(h_df)) / float(max(len(t_df),len(l_df),len(h_df)))
    if size_similarity < 0.9:
        print "WARNING: ONLY %.1f%% TARGETED MESSAGES FOR ALL TRACKED MESSAGES" % (size_similarity*100)

    pool_df = pd.concat([t_df, l_df, h_df], axis=1).fillna(method='ffill')
    return pool_df

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

    #-ve offset means calculated over the whole time
    npts = df['in_area'].sum()
    #percentage of time in area
    total_pct = 100.0 * (npts / float(len(df)))
    offset.append(-1)
    pct.append(total_pct)

    return offset, pct

def calculate_latency_to_stay(tdf, holdtime=20, minlen=10000):
    tts = []

    for name, group in tdf.groupby('obj_id'):
        if len(group) < minlen:
            print "\tlatency: skipping obj_id", name, "too short", len(group)
            continue

        t0 = group.head(1)
        if t0['in_area']:
            print "\tlatency: skipping obj_id", name, "already in area"
            continue

        #timestamp of experiment start
        t00 = t0 = t1 = t0.index[0].asm8.astype(np.int64) / 1e9

        t_in_area = 0
        for ix,row in group.iterrows():

            t1 = ix.asm8.astype(np.int64) / 1e9
            dt = t1 - t0

            if row['in_area']:
                t_in_area += dt
                if t_in_area > holdtime:
                    break

            t0 = t1

        #did the fly finish inside, and start outside
        if row['in_area'] and (t_in_area > holdtime):
            tts.append( t1 - t00 )

    return tts


