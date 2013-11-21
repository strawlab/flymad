import json
import os.path
import datetime
import math
import time
import collections
import tempfile
import shutil
import itertools

import sh
import cv2
import pytz
import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import progressbar
import adskalman.adskalman

import motmot.FlyMovieFormat.FlyMovieFormat
import benu.benu
import benu.utils

import roslib; roslib.load_manifest('rosbag')
import rosbag

assert benu.__version__ >= "0.1.0"

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

class Kalman:
    ### KEEP THESE IN SYNC WITH FLYMAD TRACKER

    FPS = 100
    Qsigma=10.0 # process covariance
    Rsigma=10.0 # observation covariance

    dt = 1.0/FPS

    # process model
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]],
                 dtype=np.float64)
    # observation model
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]],
                 dtype=np.float64)
    # process covariance
    Q = Qsigma*np.eye(4)
    # measurement covariance
    R = Rsigma*np.eye(2)

    def smooth(self, x, y):

        y = np.c_[x,y]
        initx = np.array([y[0,0],y[0,1],0,0])
        initV = 0*np.eye(4)

        xsmooth,Vsmooth = adskalman.adskalman.kalman_smoother(y,
                                self.A,self.C,
                                self.Q,self.R,
                                initx,initV)

        return xsmooth

class Arena:

    CONVERT_OPTIONS = {
        False:None,
        "m":1.0,
        "cm":100.0,
        "mm":1000.0,
    }

    def __init__(self, convert, jsonconf=dict()):
        x = jsonconf.get('cx',360)
        y = jsonconf.get('cy',255)
        r = jsonconf.get('cr',200)
        xlim = jsonconf.get('xlim',(150,570))
        ylim = jsonconf.get('ylim',(47,463))
        self._x, self._y, self._r = x,y,r
        self._xlim = xlim
        self._ylim = ylim
        self._convert = convert
        self._convert_mult = self.CONVERT_OPTIONS[convert]
        self._rw = float(jsonconf.get('rw',0.045))   #radius in m
        self._sx = float(jsonconf.get('sx',0.045/208)) #scale factor px->m
        self._sy = float(jsonconf.get('sy',0.045/219)) #scale factor px->m

        #cache the simgear object for quick tests if the fly is in the area
        (sgcx,sgcy),sgr = self.circ
        self._sg_circ = sg.Point(sgcx,sgcy).buffer(sgr)

    @property
    def unit(self):
        if self._convert:
            return self._convert
        else:
            return 'px'
    @property
    def circ(self):
        if self._convert:
            return (0,0), self._convert_mult*self._rw
        return (self._x,self._y), self._r
    @property
    def cx(self):
        return self.scale_x(self._x)
    @property
    def cy(self):
        return self.scale_y(self._y)
    @property
    def r(self):
        if self._convert:
            return self._convert_mult*self._rw
        return self._r
    @property
    def xlim(self):
        if self._convert:
            #10% larger than the radius
            xlim = np.array([-self._rw, self._rw])*self._convert_mult
            return (xlim + (xlim*0.1)).tolist()
        return self._xlim
    @property
    def ylim(self):
        if self._convert:
            #10% larger than the radius
            ylim = np.array([-self._rw, self._rw])*self._convert_mult
            return (ylim + (ylim*0.1)).tolist()
        return self._ylim

    def scale_x(self, x, origin=None):
        if self._convert:
            if origin is not None:
                o = origin
            else:
                o = self._x
            if isinstance(x,list):
                x = np.array(x)
            return (x-o)*self._sx*self._convert_mult
        return x

    def scale_y(self, y, origin=None):
        if self._convert:
            if origin is not None:
                o = origin
            else:
                o = self._y
            if isinstance(y,list):
                y = np.array(y)
            return (y-o)*self._sy*self._convert_mult
        return y

    def scale_vx(self, x):
        #when scaling velocity, don't adjust for the origin
        return self.scale_x(x, origin=0.0)

    def scale_vy(self, y):
        #when scaling velocity, don't adjust for the origin
        return self.scale_y(y, origin=0.0)

    def scale(self, v):
        if self._convert:
            if self._sx == self._sy:
                s = self._sx
            else:
                print "warning: x and y scale not identical"
                s = (self._sx + self._sy) / 2.0
            if isinstance(v,list):
                v = np.array(v)
            return v*s*self._convert_mult
        else:
            return v

    def get_intersect_polygon(self, geom):
        if self._convert and geom:
            points_x, points_y = geom
            geom = (map(self.scale_x,points_x),map(self.scale_y,points_y))

        if geom:
            poly = sg.Polygon(list(zip(*geom)))
            inter = self._sg_circ.intersection(poly)
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
        (cx,cy),r = self.circ
        return matplotlib.patches.Circle((cx,cy), radius=r, **kwargs)

    def get_limits(self):
        #(xlim, ylim)
        return self.xlim, self.ylim

def strptime_bagfile(bag):
    assert os.path.isfile(bag)
    BAG_DATE_FMT = "%Y-%m-%d-%H-%M-%S.bag"
    return time.strptime(os.path.basename(bag), BAG_DATE_FMT)

def colors_hsv_circle(n, alpha=1.0):
    _hsv = np.dstack( (np.linspace(0,2/3.,n), [1]*n, [1]*n) )
    _rgb = matplotlib.colors.hsv_to_rgb(_hsv)
    return np.dstack((_rgb, [alpha]*n))[0]

def get_path(path, dat, bname):
    bpath = dat.get('_base',os.path.abspath(os.path.dirname(path)))
    return os.path.join(bpath, bname)

def plot_geom(ax, geom):
    ax.plot(geom[0],geom[1],'g-')

def plot_laser_trajectory(ax, df, arena, plot_starts=False, plot_laser=False, intersect_patch=None):
    ax.set_aspect('equal')

    xlim,ylim = arena.get_limits()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if intersect_patch is not None:
        ax.add_patch(intersect_patch)

    first = True
    for name, group in df.groupby('lobj_id'):
        fly_x = group['fly_x'].values
        fly_y = group['fly_y'].values

        pp = ax.plot(fly_x, fly_y,
                     'k.',label="predicted" if first else "__nolegend__")

#        ax.plot(fly_x[0],fly_y[0],'b.',label="predicted" if first else "__nolegend__")

        #plot the laser when under fine control
        laserdf = group[group['mode'] == 2]
        lp = ax.plot(laserdf['laser_x'].values,
                     laserdf['laser_y'].values,
                     'r.',label="required" if first else "__nolegend__")

        first = False

def plot_tracked_trajectory(ax, tdf, arena, ds=1, minlenpct=0.10, debug_plot=True, title='', **kwargs):
    ax.set_aspect('equal')

    if debug_plot:
        outs = {}
        ins = {}

    pcts = {}

    xlim,ylim = arena.get_limits()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for experiment,df in tdf.groupby('experiment'):
        print "\ttraj: EXPERIMENT #",experiment

        for name, group in df.groupby('tobj_id'):
            lenpct = len(group) / float(len(df))
            pcts[name] = lenpct

            if lenpct < minlenpct:
                print "\ttraj: skipping obj_id", name, "len", lenpct
                if debug_plot:
                    outs[name] = (group['x'].values.copy(),group['y'].values.copy())
                continue

            print "\ttraj: obj_id", name, "len", lenpct
            ax.plot(group['x'].values[::ds], group['y'].values[::ds], **kwargs)

            if debug_plot:
                ins[name] = (group['x'].values.copy(),group['y'].values.copy())

    if debug_plot:
        ax = plt.figure("P%se%s" % (title,experiment), figsize=(6,4)).gca()
        for oid in outs:
            x,y = outs[oid]
            ax.plot(x,y,'b,',label="o%s (%.1f%%)" % (oid,pcts[oid]*100))

        for oid in ins:
            x,y = ins[oid]
            ax.plot(x, y, 'r,',label="i%s (%.1f%%)" % (oid,pcts[oid]*100))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', ncol=1, numpoints=1, markerscale=3, columnspacing=1, bbox_to_anchor=(1, 0.5), prop={'size':9})

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        patch = arena.get_patch(fill=False, color='k')
        ax.add_patch(patch)


def get_offset_and_nearest_fmf_timestamp(tss, timestamp):
    at_or_before_timestamp_cond = tss <= timestamp
    nz = np.nonzero(at_or_before_timestamp_cond)[0]
    if len(nz)==0:
        raise ValueError("no frames at or before timestamp given")
    return nz[-1], tss[nz[-1]]

def get_framedf(df, framenumber):
    return df[df['t_framenumber'] == framenumber]

def merge_bagfiles(bfs, geom_must_interect=True):
    expn = 0
    l_df, t_df, h_df, geom = bfs[0]
    t_df['experiment'] = expn

    if len(bfs) == 1:
        print "merge not needed"
        return l_df, t_df, h_df, geom

    for (_l_df, _t_df, _h_df, _geom) in bfs[1:]:
        expn += 1
        _t_df['experiment'] = expn

        l_df = l_df.append(_l_df, verify_integrity=True)
        t_df = t_df.append(_t_df, verify_integrity=True)
        h_df = h_df.append(_h_df, verify_integrity=True)

        #check this is the same trial
        oldgp = sg.Polygon(list(zip(*_geom)))
        newgp = sg.Polygon(list(zip(*geom)))
        if not oldgp.intersects(newgp):
            print "warning: geometry does not intersect", oldgp, "vs", newgp
            if geom_must_interect:
                raise ValueError("Cool tile geometry does not intersect "\
                                 "(set json _geom_must_intersect key to "\
                                 "false to override")

    return l_df, t_df, h_df, geom

def load_bagfile(bagpath, arena, filter_short=100, filter_short_pct=0, smooth=False):
    def in_area(row, poly):
        if poly:
            in_area = poly.contains( sg.Point(row['x'], row['y']) )
        else:
            in_area = False
        return pd.Series({"in_area":in_area})

    print "loading", bagpath
    bag = rosbag.Bag(bagpath)

    geom_msg = None

    #KEEP TRACKED AND LASER OBJECT ID SEPARATE

    l_index = []
    l_data = {k:[] for k in ("lobj_id",
                             "fly_x","fly_y","laser_x","laser_y",
                             "fly_x_px","fly_y_px","laser_x_px","laser_y_px",
                             "laser_power","mode")}

    t_index = []
    t_data = {k:[] for k in ("tobj_id","x_px","y_px","vx_px","vy_px","v_px",'t_framenumber')}

    h_index = []
    h_data = {k:[] for k in ("head_x", "head_y", "body_x", "body_y", "target_x", "target_y", "target_type", "h_framenumber", "h_processing_time")}
    h_data_names = ("head_x", "head_y", "body_x", "body_y", "target_x", "target_y", "target_type")

    r_data = {k:[] for k in ("r_framenumber", "theta")}

    for topic,msg,rostime in bag.read_messages(topics=["/targeter/targeted",
                                                       "/flymad/tracked",
                                                       "/draw_geom/poly",
                                                       "/flymad/laser_head_delta",
                                                       "/flymad/raw_2d_positions"]):
        if topic == "/targeter/targeted":
            l_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
            l_data['lobj_id'].append(msg.obj_id)
            l_data['laser_power'].append(msg.laser_power)
            l_data['mode'].append(msg.mode)
            l_data['fly_x_px'].append(msg.fly_x)
            l_data['fly_y_px'].append(msg.fly_y)
            l_data['laser_x_px'].append(msg.laser_x)
            l_data['laser_y_px'].append(msg.laser_y)
        elif topic == "/flymad/tracked":
            if msg.is_living:
                t_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
                t_data['tobj_id'].append(msg.obj_id)
                t_data['t_framenumber'].append(msg.framenumber)
                t_data['x_px'].append(msg.state_vec[0])
                t_data['y_px'].append(msg.state_vec[1])
                t_data['vx_px'].append(msg.state_vec[2])
                t_data['vy_px'].append(msg.state_vec[3])
        elif topic == "/draw_geom/poly":
            if geom_msg is not None:
                print "WARNING: DUPLICATE GEOM MSG", msg, "vs", geom_msg
            geom_msg = msg
        elif topic == "/flymad/laser_head_delta":
            h_index.append( datetime.datetime.fromtimestamp(rostime.to_sec()) )
            for k in h_data_names:
                h_data[k].append( getattr(msg,k,np.nan) )
            h_data["h_framenumber"].append( getattr(msg,"framenumber",0) )
            h_data["h_processing_time"].append( msg.processing_time )
        elif topic == "/flymad/raw_2d_positions":
            if len(msg.points) == 1:
                r_data["r_framenumber"].append(msg.framenumber)
                r_data["theta"].append(msg.points[0].theta)

    if geom_msg is not None:
        points_x = [pt.x for pt in geom_msg.points]
        points_y = [pt.y for pt in geom_msg.points]
        geom = (points_x, points_y)
    else:
        geom = tuple()

    if smooth:
        print "\tkalman smoothing"

        dt = 1/100.0#np.gradient(df.index.values.astype('float64')/SECOND_TO_NANOSEC)

        #smooth the positions, and recalculate the velocitys based on this.
        kf = Kalman()
        smoothed = kf.smooth(t_data['x_px'], t_data['y_px'])
        x_px = smoothed[:,0]
        y_px = smoothed[:,1]
        vx_px = np.gradient(x_px) / dt
        vy_px = np.gradient(y_px) / dt
    else:
        x_px = np.array(t_data['x_px'])
        y_px = np.array(t_data['y_px'])
        vx_px = np.array(t_data['vx_px'])
        vy_px = np.array(t_data['vy_px'])

    #convert to real world units if the arena supports it
    #KEEP THIS UPDATED TO INCLUDE ALL PIXEL FIELDS IN THE BAGFILES
    t_data['v_px'] = np.sqrt((vx_px**2) + (vy_px**2))

    t_data['x'] = arena.scale_x(x_px)
    t_data['y'] = arena.scale_y(y_px)
    t_data['vx'] = arena.scale_vx(vx_px)
    t_data['vy'] = arena.scale_vy(vy_px)
    t_data['v'] = np.sqrt((t_data['vx']**2) + (t_data['vy']**2))

    l_data["fly_x"] = arena.scale_x(l_data["fly_x_px"])
    l_data["fly_y"] = arena.scale_y(l_data["fly_y_px"])
    l_data["laser_x"] = arena.scale_x(l_data["laser_x_px"])
    l_data["laser_y"] = arena.scale_y(l_data["laser_y_px"])

    poly = arena.get_intersect_polygon(geom)

    l_df = pd.DataFrame(l_data, index=l_index)
    t_df = pd.DataFrame(t_data, index=t_index)
    h_df = pd.DataFrame(h_data, index=h_index)

    r_df = pd.DataFrame(r_data)
    if len(r_df) == len(t_df):
        t_df['theta'] = r_df['theta'].values
    else:
        t_df['theta'] = np.nan

    #add a new colum if they were in the area
    t_df = pd.concat([t_df,
                      t_df.apply(in_area, axis=1, args=(poly,))],
                      axis=1)

    if (filter_short_pct > 0) or (filter_short > 0):
        if filter_short_pct > 0:
            filter_short = (float(filter_short_pct)/100.0) * len(t_df)


    #optionally find short trials here, print the length of kept trials
    short_tracks = []
    for name, group in t_df.groupby('tobj_id'):
        if len(group) < filter_short:
            print '\tload: skip trajectory obj_id %s (%s (%.1f%%) long)' % (
                            name, len(group), 100.0*len(group)/len(t_df))
            short_tracks.append(name)
        else:
            print '\tload: trajectory obj_id %s (%s (%.1f%%) long)' % (
                            name, len(group), 100.0*len(group)/len(t_df))

    if short_tracks:
        l_df = l_df[~l_df['lobj_id'].isin(short_tracks)]
        t_df = t_df[~t_df['tobj_id'].isin(short_tracks)]

    t_df['experiment'] = 0

    return l_df, t_df, h_df, geom

def load_bagfile_single_dataframe(bagpath, arena, ffill, warn=False, **kwargs):
    l_df, t_df, h_df, geom = load_bagfile(bagpath, arena, **kwargs)

    #merge the dataframes
    if warn:
        #check we have about the same amount of data
        size_similarity = min(len(t_df),len(l_df),len(h_df)) / float(max(len(t_df),len(l_df),len(h_df)))
        if size_similarity < 0.9:
            print "WARNING: ONLY %.1f%% TARGETED MESSAGES FOR ALL TRACKED MESSAGES" % (size_similarity*100)

    pool_df = pd.concat([t_df, l_df, h_df], axis=1)
    if ffill is True:
        return pool_df.fillna(method='ffill')
    elif isinstance(ffill, list) or isinstance(ffill, tuple):
        for col in ffill:
            pool_df[col].fillna(method='ffill', inplace=True)

    return pool_df

def calculate_total_pct_in_area(tdf, maxtime):
    pcts = []

    for experiment,df in tdf.groupby('experiment'):
        print "\ttpct: EXPERIMENT #",experiment

        t00 = df.index[0]
        tend = min(t00 + datetime.timedelta(seconds=maxtime), df.index[-1])

        npts = df['in_area'].sum()
        #percentage of time in area
        total_pct = 100.0 * (npts / float(len(df)))

        pcts.append(total_pct)

    return pcts

def calculate_time_in_area(tdf, maxtime, toffsets):
    exp_pcts = []

    def _get_time_in_area(_df, _t0, _t1):
        print "\ttime: group",_t0,":",_t1

        #get trajectories of that timespan
        idf = _df[_t0:_t1]
        #number of timepoints in the area
        npts = idf['in_area'].sum()
        #percentage of time in area
        try:
            pct = 100.0 * (npts / float(len(idf)))
        except ZeroDivisionError:
            #short dataframe - see below
            pct = 0
        return pct

    for experiment,df in tdf.groupby('experiment'):
        print "\ttime: EXPERIMENT #",experiment
        pcts = []

        #put the df into 10ms bins
        #df = tdf.resample('10L')

        #maybe pandas has a built in way to do this?
        t00 = df.index[0]
        tend = min(t00 + datetime.timedelta(seconds=maxtime), df.index[-1])

        for offset0,offset1 in pairwise(toffsets):
            t0 = t00 + datetime.timedelta(seconds=offset0)
            t1 = t00 + datetime.timedelta(seconds=offset1)

            if t1 > tend:
                print "WARNING: DATAFRAME SHORTER THAN MAX TIME",maxtime,"ONLY", df.index[-1] - df.index[0],"LONG"

            pcts.append( _get_time_in_area(df,t0,t1) )

        #the last time period
        pcts.append( _get_time_in_area(df,t1,tend) )

        exp_pcts.append(pcts)

    #rows are experiments, cols are the time bins
    return np.r_[exp_pcts]

def calculate_latency_and_velocity_to_stay(tdf, holdtime=20, minlenpct=0.10, tout_reset_time=1, arena=None, geom=None, debug_plot=True, title=''):
    tts = []
    vel_out = []
    vel_in = []

    for experiment,df in tdf.groupby('experiment'):
        print "\tltcy: EXPERIMENT #",experiment
        for name, group in df.groupby('tobj_id'):
            lenpct = len(group) / float(len(df))
            if lenpct < minlenpct:
                print "\tltcy: skipping obj_id", name, "len", lenpct
                continue

            t0 = group.head(1)
            if t0['in_area']:
                print "\tltcy: skipping obj_id", name, "already in area"
                continue

            print "\tltcy: obj_id", name, "len", lenpct

            #timestamp of experiment start
            t0ix = t0.index[0]
            t00 = t0 = t1 = t0ix.asm8.astype(np.int64) / 1e9

            t_in_area = 0
            t_out_area = 0
            for ix,row in group.iterrows():

                t1 = ix.asm8.astype(np.int64) / 1e9
                dt = t1 - t0

                if row['in_area']:
                    t_out_area = 0
                    t_in_area += dt
                    if t_in_area > holdtime:
                        #the fly made it
                        break
                else:
                    t_out_area += dt
                    if (t_out_area > tout_reset_time):
                        t_in_area = 0

                t0 = t1

            #this is either the time for the fly to make it, or the total time of
            #the experiment
            tts.append( t1 - t00 )

            #if the fly finished inside then slicing the trajectory into parts
            #inside and outside makes sense.
            if row['in_area'] and (t_in_area > holdtime):
                print "\tltcy: obj_id %s finished inside after %.1f (in for %.1f)" % (name, tts[-1], t_in_area)
                #the time they first got to the area, more or less beucause there
                #could is tout_reset_time hysteresis is the most recent index minus
                #the t_in_area
                t_first_in_area_ix = ix - datetime.timedelta(seconds=t_in_area)
                t_last_in_area = ix

                dfo = group[:t_first_in_area_ix]
                dfi = group[t_first_in_area_ix:t_last_in_area]

                vel_out.append( dfo['v'].mean() )
                vel_in.append( dfi['v'].mean() )

                if debug_plot:

                    xlim,ylim = arena.get_limits()

                    ax = plt.figure("V%se%s oid:%s" % (title,experiment,name)).gca()
                    ax.plot(dfo.index.astype(np.int64)/1e9, dfo['v'].values, 'b')
                    ax.plot(dfi.index.astype(np.int64)/1e9, dfi['v'].values, 'r')

                    ax = plt.figure("P%se%s oid:%s" % (title,experiment,name)).gca()
                    ax.plot(dfo['x'],dfo['y'],'b,')
                    ax.plot(dfi['x'],dfi['y'],'r,')
                    ax.plot(group['x'].values[0],group['y'].values[0],'go')
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)

                    patch = arena.get_intersect_patch(geom, fill=True, color='r', closed=True, alpha=0.2)
                    ax.add_patch(patch)
                    patch = arena.get_patch(fill=False, color='k')
                    ax.add_patch(patch)

            else:
                #the fly didn't make it, so consider the whole trajectory
                #for its outside velocity
                vel_out.append( group['v'].mean() )
                print "\tltcy: obj_id %s finished outside" % name

    return tts, vel_out, vel_in

def get_progress_bar(name, maxval):
    widgets = ["%s: " % name, progressbar.Percentage(),
               progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,maxval=maxval).start()
    return pbar

FMFFrame = collections.namedtuple('FMFFrame', 'offset timestamp')

class FrameDescriptor:
    def __init__(self, w_frame, z_frame, df_or_series, epoch):
        self.w_frame = w_frame
        self.z_frame = z_frame
        self.df = df_or_series
        self.epoch = epoch

    def get_row(self, *cols):
        if isinstance(self.df, pd.Series):
            return self.df
        elif isinstance(self.df, pd.DataFrame):
            #return the most recent row always
            return self.df.dropna(subset=cols).tail(1)

class _FMFPlotter:

    t0 = 0
    force_color = False
    alpha = None
    beta = None

    width = 659
    height = 494

    show_timestamp = True
    show_epoch = True
    show_lxly = False
    show_fxfy = True

    show_arena = False

    def __init__(self, path):
        if path:
            self.fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(path)
            self.width = self.fmf.width
            self.height = self.fmf.height
        else:
            self.fmf = None

    def enable_show_arena(self, arena):
        self.show_arena = arena

    def enable_force_rgb(self):
        self.force_color = True

    def enable_color_correction(self, brightness, contrast):
        assert 0 < brightness < 100
        assert 1.0 <= contrast <= 3.0
        self.alpha = contrast
        self.beta = brightness

    def get_frame(self, frame):
        assert isinstance(frame, FMFFrame)

        if self.fmf is None:
            return None

        f,ts = self.fmf.get_frame(frame.offset)

        assert ts == frame.timestamp

        if self.force_color:
            return cv2.cvtColor(f,cv2.COLOR_GRAY2RGB)
        elif (self.alpha is not None) and (self.beta is not None):
            mul_f = cv2.multiply(f,np.array([self.alpha],dtype=float))
            return cv2.add(mul_f,np.array([self.beta],dtype=float))
        else:
            return f

    def get_benu_panel(self, device_x0, device_x1, device_y0, device_y1):
        return dict(
            width = self.width,
            height = self.height,
            device_x0 = device_x0,
            device_x1 = device_x1,
            device_y0 = device_y0,
            device_y1 = device_y1,
        )

    def imshow(self, canv, frame):
        if (self.fmf is None) or (frame is None):
            canv.poly([0,0,self.width,self.width,0],
                      [0,self.height,self.height,0,0],
                      color_rgba=(0,0,0,1))
        else:
            canv.imshow(self.get_frame(frame), 0,0, filter='best')

        if self.show_arena:
            canv.scatter( [self.show_arena.cx],
                          [self.show_arena.cy],
                          radius=self.show_arena.r )

class ArenaPlotter(_FMFPlotter):

    show_theta = False
    show_velocity = False
    show_framenumber = False

    def __init__(self, arena, bgcolor=(0.0,0.0,0.0,1), tzname='CET'):
        _FMFPlotter.__init__(self, None) #no fmf
        self.bgcolor = bgcolor
        self.tz = pytz.timezone( tzname )
        self.enable_show_arena(arena)

    def render(self, canv, panel, desc):
        with canv.set_user_coords_from_panel(panel):
            #draw the arena
            self.imshow(canv, None)

            row = desc.get_row()

            if self.show_fxfy:
                canv.scatter( [row['x']],
                              [row['y']],
                              color_rgba=(0,1,0,1),
                              radius=2 )

            if self.show_theta and row['theta']:
                try:
                    theta = row['theta'].values[0]
                    fx = float(row['x'])
                    fy = float(row['y'])
                except:
                    theta = fx = fy = np.nan

                if not np.isnan(theta):
                    xarr = [fx-5, fx + 5]
                    yarr = [fy-(5*np.arctan(theta)), fy+(5*np.arctan(theta))]
                    canv.plot(xarr,yarr,
                            color_rgba=(0,1,0,1),
                            close_path=False)

            if self.show_lxly and row['laser_power']:
                canv.scatter( [row['laser_x']],
                              [row['laser_y']],
                              color_rgba=(1,0,0,1),
                              radius=1 )

            if self.show_epoch:
                canv.text(str(datetime.datetime.fromtimestamp(desc.epoch, self.tz)),
                          15,25,
                          color_rgba=(1.,1.,1.,1.),
                          font_face="Ubuntu", bold=False, font_size=14)

            if self.show_framenumber:
                canv.text(str(int(row['t_framenumber'])),
                          15,50,
                          color_rgba=(1.,1.,1.,1.),
                          font_face="Ubuntu", bold=False, font_size=14)

            if self.show_epoch:
                canv.text("%.1fs" % (desc.epoch - self.t0),
                          15,75,
                          color_rgba=(1.,1.,1.,1.),
                          font_face="Ubuntu", bold=False, font_size=14)

            if self.show_velocity:
                canv.text("%.1f px/s" % row['v'],
                          15,panel["height"]-15,
                          color_rgba=(1.,1.,1.,1.),
                          font_face="Ubuntu", bold=False, font_size=14)


class FMFImagePlotter(_FMFPlotter):

    def __init__(self, path, framename):
        _FMFPlotter.__init__(self, path)
        self.name = framename[0]
        self._framename = framename

    def render(self, canv, panel, desc):
        with canv.set_user_coords_from_panel(panel):
            self.imshow(canv, getattr(desc,self._framename))

class _MultiTrajectoryColorManager:
    def __init__(self, objids):
        #ignore the red color (color 0), we use it for target indication
        n_objids = len(objids)
        colors = colors_hsv_circle(n_objids+1)[1:]

        self.trajs_colors = {}
        for oid,col in zip(objids,colors):
            self.trajs_colors[int(oid)] = col

class FMFMultiTrajectoryPlotter(_FMFPlotter, _MultiTrajectoryColorManager):

    name = 'w'

    def __init__(self, path, objids,maxlen=100):
        _FMFPlotter.__init__(self, path)
        _MultiTrajectoryColorManager.__init__(self, objids)
        self.trajs_x = {}
        self.trajs_y = {}
        self.trajs_last_seen = {}
        self.maxlen=maxlen

    def render(self, canv, panel, desc):
        w_framenumber = desc.w_frame.timestamp
        to_kill = []

        #get the targeted fly
        rowt = desc.get_row('lobj_id','fly_x', 'fly_y')

        #birth and update the trajectory history of all previous flies
        for oid,_row in desc.df.groupby('tobj_id'):
            row = _row.tail(1)
            t_framenumber = row['t_framenumber'].values[0]
            if oid not in self.trajs_x:
                self.trajs_x[oid] = collections.deque(maxlen=self.maxlen)
                self.trajs_y[oid] = collections.deque(maxlen=self.maxlen)

            self.trajs_x[oid].append(row['x'])
            self.trajs_y[oid].append(row['y'])
            self.trajs_last_seen[oid] = t_framenumber

        with canv.set_user_coords_from_panel(panel):
            self.imshow(canv, desc.w_frame)

            #draw all trajectories
            for oid in self.trajs_x:

                #check for old, dead trajectories
                if (w_framenumber - self.trajs_last_seen[oid]) > 5:
                    to_kill.append(oid)
                    continue

                canv.scatter( self.trajs_x[oid],
                              self.trajs_y[oid],
                              color_rgba=self.trajs_colors[int(oid)], radius=0.5 )

            #draw the targeted fly (if during this frame interval we targeted
            #a single fly only)
            if self.show_fxfy and (len(rowt) == 1):
                canv.scatter( [rowt['fly_x']],
                              [rowt['fly_y']],
                              color_rgba=(1,0,0,0.3), radius=10.0 )

            if self.show_timestamp:
                canv.text(str(int(desc.w_frame.timestamp)),
                          panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            if self.show_epoch:
                canv.text("%.1fs" % (desc.epoch - self.t0),
                          panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))

        for oid in to_kill:
            del self.trajs_x[oid]
            del self.trajs_y[oid]

class FMFTrajectoryPlotter(_FMFPlotter):

    name = 'w'

    def __init__(self, path,maxlen=100):
        _FMFPlotter.__init__(self, path)
        self.xhist = collections.deque(maxlen=maxlen)
        self.yhist = collections.deque(maxlen=maxlen)

    def render(self, canv, panel, desc):
        row = desc.get_row('fly_x', 'fly_y', 'laser_x', 'laser_y', 'mode')

        x,y = row['fly_x'],row['fly_y']
        lx,ly,mode = row['laser_x'],row['laser_y'],row['mode']

        self.xhist.append(x)
        self.yhist.append(y)

        with canv.set_user_coords_from_panel(panel):
            self.imshow(canv, desc.w_frame)

            canv.scatter( self.xhist,
                          self.yhist,
                          color_rgba=(0,1,0,0.3), radius=0.5 )

            if self.show_fxfy:
                canv.scatter( [x],
                              [y],
                              color_rgba=(1,0,0,0.3), radius=10.0 )

            if self.show_lxly and (mode == 2):
                canv.scatter( [lx],
                              [ly],
                              color_rgba=(0,0,1,0.3), radius=2.0 )

            if self.show_timestamp:
                canv.text(str(int(desc.w_frame.timestamp)),
                          panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            if self.show_epoch:
                canv.text("%.1fs" % (desc.epoch - self.t0),
                          panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))


class FMFTTLPlotter(_FMFPlotter):

    name = 'f'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.hx = self.hy = 0

    def get_fly_color(self, oid):
        return (0,1,0,0.3)

    def render(self, canv, panel, desc):
        row = desc.get_row('head_x','head_y','target_x','target_y')

        rowl = desc.get_row('mode','lobj_id')
        mode_s = row_to_target_mode_string(rowl)

        hx,hy = row['head_x'],row['head_y']
        tx,ty = row['target_x'],row['target_y']

        with canv.set_user_coords_from_panel(panel):
            self.imshow(canv, desc.z_frame)

            canv.scatter( [tx],
                          [ty],
                          color_rgba=(1,0,0,0.3), radius=5.0 )

            if mode_s != NO_TARGET_STRING:
                canv.scatter( [hx],
                              [hy],
                              color_rgba=self.get_fly_color(rowl['lobj_id']), radius=10.0 )

            if self.show_timestamp:
                canv.text(str(int(desc.z_frame.timestamp)),
                          panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            canv.text(mode_s,
                      panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))

class FMFMultiTTLPlotter(FMFTTLPlotter, _MultiTrajectoryColorManager):

    def __init__(self, path, objids):
        FMFTTLPlotter.__init__(self, path)
        _MultiTrajectoryColorManager.__init__(self, objids)

    def get_fly_color(self, oid):
        try:
            return self.trajs_colors[int(oid)]
        except (KeyError, ValueError):
            return (0,0,0,0.3)


### keep in sync with refined_utils.py
def target_dx_dy_from_message(row):
    """returns None,None if the head/body was not detected"""

    dx = dy = np.nan
    tx = ty = 1e6

    if row['target_type'] == 1:
        tx = row['head_x']
        ty = row['head_y']
    elif row['target_type'] == 2:
        tx = row['body_x']
        ty = row['body_y']

    if tx != 1e6:
        dx = row['target_x'] - tx
        dy = row['target_y'] - ty

    return dx, dy

NO_TARGET_STRING = "Idle"

def row_to_target_mode_string(row):
    if row['mode'] == 1:
        return "Wide"
    elif row['mode'] == 2:
        return "TTM"
    else:
        return NO_TARGET_STRING

class TTLPlotter:

    HIST = 3.0

    def __init__(self, t0, ifi):
        self.t0 = t0
        self.ifi = ifi
        maxlen = self.HIST / self.ifi
        self.dx = collections.deque(maxlen=maxlen)
        self.dy = collections.deque(maxlen=maxlen)

    def render(self, canv, panel, desc):
        row = desc.get_row('target_type', 'head_x', 'head_y', 'body_x', 'body_y', 'target_x', 'target_y')

        head_dx, head_dy = target_dx_dy_from_message(row)

        self.dx.appendleft(head_dx)
        self.dy.appendleft(head_dy)

        time = np.array(range(len(self.dx)))*self.ifi*-1.0

        with canv.set_user_coords_from_panel(panel):
            with canv.get_figure(panel) as fig:
                ax = fig.gca()
                ax.plot( time, self.dx, 'r-' )
                ax.plot( time, self.dy, 'b-' )
                ax.set_ylim([-200,200])
                ax.set_xlim([0,-1.0*self.HIST])
                ax.axhline(0, color='k')
                ax.yaxis.set_ticks([-150, -100, -50, 0, 50, 100, 150])
                benu.utils.set_foregroundcolor(ax, 'white')
                benu.utils.set_backgroundcolor(ax, 'black')
                fig.patch.set_facecolor('black')
                fig.subplots_adjust(left=0.035, right=0.98)

class MovieMaker:
    def __init__(self, tmpdir='/tmp/', obj_id='movie', fps=20):
        self.tmpdir = tempfile.mkdtemp(str(obj_id), dir=tmpdir)
        self.obj_id = obj_id
        self.num = 0
        self.fps = fps

        print "movies temporary files saved to %s" % self.tmpdir

    @property
    def movie_fname(self):
        return "%s.mp4" % self.obj_id

    @property
    def frame_number(self):
        return self.num

    def next_frame(self):
        self.num += 1
        return self.new_frame(self.num)

    def new_frame(self, num):
        self.num = num
        return os.path.join(self.tmpdir,"frame{:0>6d}.png".format(num))

    def render(self, moviedir):
        sh.mplayer("mf://%s/frame*.png" % self.tmpdir,
                   "-mf", "fps=%d" % self.fps,
                   "-vo", "yuv4mpeg:file=%s/movie.y4m" % self.tmpdir,
                   "-ao", "null", 
                   "-nosound", "-noframedrop", "-benchmark", "-nolirc"
        )

        if not os.path.isdir(moviedir):
            os.makedirs(moviedir)
        moviefname = os.path.join(moviedir,"%s.mp4" % self.obj_id)

        sh.x264("--output=%s/movie.mp4" % self.tmpdir,
                "%s/movie.y4m" % self.tmpdir,
        )

        sh.mv("-u", "%s/movie.mp4" % self.tmpdir, moviefname)
        return moviefname

    def cleanup(self):
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    for unit in ('mm','cm','m'):
        a = Arena(unit)

        cx = a._x #cx
        cy = a._y #cy
        print "cx =", a.scale_x(cx), unit
        print "cy =", a.scale_y(cy), unit

        print "200 px from origin +ve x", a.scale_x(cx + a._r), unit
        print "200 px from origin +ve y", a.scale_y(cy + a._r), unit

        print "400 px/s in +ve x", a.scale_vx(400), unit+'/s'
        print "400 px/s in +ve y", a.scale_vy(400), unit+'/s'

