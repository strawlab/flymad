import json
import os.path
import datetime
import math
import collections
import tempfile
import shutil

import sh
import cv2
import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import progressbar

import motmot.FlyMovieFormat.FlyMovieFormat
import benu.benu
import benu.utils

import roslib; roslib.load_manifest('rosbag')
import rosbag

assert benu.__version__ >= "0.1.0"

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

def colors_hsv_circle(n, alpha=1.0):
    _hsv = np.dstack( (np.linspace(0,2/3.,n), [1]*n, [1]*n) )
    _rgb = matplotlib.colors.hsv_to_rgb(_hsv)
    return np.dstack((_rgb, [alpha]*n))[0]

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
    for name, group in df.groupby('lobj_id'):
        fly_x = group['fly_x'].values
        fly_y = group['fly_y'].values

        pp = ax.plot(fly_x,fly_y,'k.',label="predicted" if first else "__nolegend__")

#        ax.plot(fly_x[0],fly_y[0],'b.',label="predicted" if first else "__nolegend__")

        #plot the laser when under fine control
        laserdf = group[group['mode'] == 2]
        lp = ax.plot(laserdf['laser_x'],laserdf['laser_y'],'r.',label="required" if first else "__nolegend__")

        first = False

def plot_tracked_trajectory(ax, df, limits=None, ds=1, minlenpct=0.10, **kwargs):
    ax.set_aspect('equal')

    if limits is not None:
        xlim,ylim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for name, group in df.groupby('tobj_id'):
        lenpct = len(group) / float(len(df))
        if lenpct < minlenpct:
            print "\tskip: skipping obj_id", name, "len", lenpct
            continue

        print "\ttraj: obj_id", name, "len", lenpct

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

    #KEEP TRACKED AND LASER OBJECT ID SEPARATE

    l_index = []
    l_data = {k:[] for k in ("lobj_id","fly_x","fly_y","laser_x","laser_y","laser_power","mode")}
    l_data_names = ("fly_x","fly_y","laser_x","laser_y","laser_power","mode")

    t_index = []
    t_data = {k:[] for k in ("tobj_id","x","y","vx","vy",'v','t_framenumber')}

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
            l_data['lobj_id'].append(msg.obj_id)
        elif topic == "/flymad/tracked":
            if msg.is_living:
                vx = msg.state_vec[2]
                vy = msg.state_vec[3]
                t_index.append( datetime.datetime.fromtimestamp(msg.header.stamp.to_sec()) )
                t_data['tobj_id'].append(msg.obj_id)
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
                h_data[k].append( getattr(msg,k,np.nan) )
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
        for name, group in t_df.groupby('tobj_id'):
            if len(group) < filter_short:
                print '\tload ignoring trajectory with obj_id %s (%s samples long)' % (name, len(group))
                short_tracks.append(name)

        l_df = l_df[~l_df['lobj_id'].isin(short_tracks)]
        t_df = t_df[~t_df['tobj_id'].isin(short_tracks)]

    return l_df, t_df, h_df, geom

def load_bagfile_single_dataframe(bagpath, arena, ffill, **kwargs):
    l_df, t_df, h_df, geom = load_bagfile(bagpath, arena, **kwargs)

    #merge the dataframes
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

def calculate_latency_to_stay(tdf, holdtime=20, minlenpct=0.10):
    tts = []

    for name, group in tdf.groupby('tobj_id'):
        lenpct = len(group) / float(len(tdf))
        if lenpct < minlenpct:
            print "\tskip: skipping obj_id", name, "len", lenpct
            continue

        t0 = group.head(1)
        if t0['in_area']:
            print "\tskip: skipping obj_id", name, "already in area"
            continue

        print "\tltcy: obj_id", name, "len", lenpct

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
            return self.df.dropna(subset=cols)

class _FMFPlotter:

    t0 = 0
    force_color = False
    alpha = None
    beta = None

    def __init__(self, path):
        self.fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(path)

    def enable_force_rgb(self):
        self.force_color = True

    def enable_color_correction(self, brightness, contrast):
        assert 0 < brightness < 100
        assert 1.0 <= contrast <= 3.0
        self.alpha = contrast
        self.beta = brightness

    def get_frame(self, frame):
        assert isinstance(frame, FMFFrame)

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
            width = self.fmf.width,
            height = self.fmf.height,
            device_x0 = device_x0,
            device_x1 = device_x1,
            device_y0 = device_y0,
            device_y1 = device_y1,
        )

class FMFImagePlotter(_FMFPlotter):

    def __init__(self, path, framename):
        _FMFPlotter.__init__(self, path)
        self.name = framename[0]
        self._framename = framename

    def render(self, canv, panel, desc):
        img = self.get_frame(getattr(desc,self._framename))
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )

class FMFMultiTrajectoryPlotter(_FMFPlotter):

    name = 'w'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.trajs_x = {}
        self.trajs_y = {}
        self.trajs_colors = {}
        self.trajs_last_seen = {}
        self.traj_n = 0

        #ignore the red color (color 0), we use it for target indication
        self._traj_color = colors_hsv_circle(10)[1:]

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
                self.trajs_x[oid] = collections.deque(maxlen=100)
                self.trajs_y[oid] = collections.deque(maxlen=100)
                self.trajs_colors[oid] = self._traj_color[self.traj_n]
                self.traj_n += 1

            self.trajs_x[oid].append(row['x'])
            self.trajs_y[oid].append(row['y'])
            self.trajs_last_seen[oid] = t_framenumber

        img = self.get_frame(desc.w_frame)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )

            #draw all trajectories
            for oid in self.trajs_colors:

                #check for old, dead trajectories
                if (w_framenumber - self.trajs_last_seen[oid]) > 5:
                    to_kill.append(oid)
                    continue

                canv.scatter( self.trajs_x[oid],
                              self.trajs_y[oid],
                              color_rgba=self.trajs_colors[oid], radius=0.5 )

            #draw the targeted fly (if during this frame interval we targeted
            #a single fly only)
            if len(rowt) == 1:
                canv.scatter( [rowt['fly_x']],
                              [rowt['fly_y']],
                              color_rgba=(1,0,0,0.3), radius=10.0 )

            canv.text(str(int(desc.w_frame.timestamp)),
                      panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            canv.text("%.1fs" % (desc.epoch - self.t0),
                      panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))

        for oid in to_kill:
            del self.trajs_colors[oid]
            del self.trajs_x[oid]
            del self.trajs_y[oid]

class FMFTrajectoryPlotter(_FMFPlotter):

    name = 'w'
    show_lxly = False
    show_fxfy = True

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.xhist = collections.deque(maxlen=100)
        self.yhist = collections.deque(maxlen=100)

    def render(self, canv, panel, desc):
        row = desc.get_row('fly_x', 'fly_y', 'laser_x', 'laser_y', 'mode')

        x,y = row['fly_x'],row['fly_y']
        lx,ly,mode = row['laser_x'],row['laser_y'],row['mode']

        self.xhist.append(x)
        self.yhist.append(y)

        img = self.get_frame(desc.w_frame)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )
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

            canv.text(str(int(desc.w_frame.timestamp)),
                      panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            canv.text("%.1fs" % (desc.epoch - self.t0),
                      panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))


class FMFTTLPlotter(_FMFPlotter):

    name = 'f'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.hx = self.hy = 0

    def render(self, canv, panel, desc):
        row = desc.get_row('head_x','head_y','target_x','target_y')

        mode_s = row_to_target_mode_string(desc.get_row('mode'))

        hx,hy = row['head_x'],row['head_y']
        tx,ty = row['target_x'],row['target_y']

        img = self.get_frame(desc.z_frame)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )
            canv.scatter( [hx],
                          [hy],
                          color_rgba=(0,1,0,0.3), radius=10.0 )
            canv.scatter( [tx],
                          [ty],
                          color_rgba=(1,0,0,0.3), radius=5.0 )

            canv.text(str(int(desc.z_frame.timestamp)),
                      panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            canv.text(mode_s,
                      panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))

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

def row_to_target_mode_string(row):
    if row['mode'] == 1:
        return "Wide"
    elif row['mode'] == 2:
        return "TTM"
    else:
        return "Idle"

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



