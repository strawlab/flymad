import sys
import collections
import os.path
import tempfile
import shutil

import cv2
import pandas as pd
import numpy as np
import sh
import progressbar

import motmot.FlyMovieFormat.FlyMovieFormat
import benu.benu
import benu.utils

import madplot

assert benu.__version__ >= "0.1.0"

MAXLEN = 100

TARGET_OUT_W, TARGET_OUT_H = 1280, 1024
MARGIN = 2

FMFFrame = collections.namedtuple('FMFFrame', 'offset timestamp')
FrameDescriptor = collections.namedtuple('FrameDescriptor', 'w_frame z_frame row epoch')

class _FMFPlotter:

    t0 = 0

    def __init__(self, path):
        self.fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(path)

    def get_frame(self, frame, color=False):
        assert isinstance(frame, FMFFrame)

        f,ts = self.fmf.get_frame(frame.offset)

        assert ts == frame.timestamp

        if color:
            return cv2.cvtColor(f,cv2.COLOR_BAYER_GR2RGB)
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

class FMFTrajectoryPlotter(_FMFPlotter):

    name = 'w'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.xhist = collections.deque(maxlen=MAXLEN)
        self.yhist = collections.deque(maxlen=MAXLEN)

    def render(self, canv, panel, desc):
        row = desc.row
        x,y = row['fly_x'],row['fly_y']

        lx,ly,mode = row['laser_x'],row['laser_y'],row['mode']

        self.xhist.append(x)
        self.yhist.append(y)

        img = self.get_frame(desc.w_frame)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )
            canv.scatter( [x],
                          [y],
                          color_rgba=(0,1,0,0.3), radius=2.0 )
            canv.scatter( self.xhist,
                          self.yhist,
                          color_rgba=(0,1,0,0.3), radius=0.5 )

            if mode == 2:
                canv.scatter( [lx],
                              [ly],
                              color_rgba=(1,0,0,0.3), radius=2.0 )

            canv.text(str(desc.w_frame.timestamp),
                      panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

            dt = desc.epoch - self.t0
            canv.text("%.1fs" % dt,panel["dw"]-40,panel["dh"]-17, color_rgba=(0.5,0.5,0.5,1.0))


class FMFTTLPlotter(_FMFPlotter):

    name = 'f'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.hx = self.hy = 0

    def render(self, canv, panel, desc):
        row = desc.row
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
                          color_rgba=(0,0,1,0.3), radius=5.0 )

            canv.text(str(desc.z_frame.timestamp),
                      panel["dw"]-40,panel["dh"]-5, color_rgba=(0.5,0.5,0.5,1.0))

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

class TTLPlotter:

    HIST = 3.0

    def __init__(self, t0, ifi):
        self.t0 = t0
        self.ifi = ifi
        maxlen = self.HIST / self.ifi
        self.dx = collections.deque(maxlen=maxlen)
        self.dy = collections.deque(maxlen=maxlen)

    def render(self, canv, panel, desc):
        head_dx, head_dy = target_dx_dy_from_message(desc.row)

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

class Assembler:
    def __init__(self, w, h, panels, wfmf, zfmf, plotttm, moviemaker):
        self.panels = panels
        self.w = w
        self.h = h
        self.wfmf = wfmf
        self.zfmf = zfmf
        self.plotttm = plotttm
        self.i = 0
        self.moviemaker = moviemaker

    def render_frame(self, desc):
        assert isinstance(desc, FrameDescriptor)

        png = self.moviemaker.next_frame()
        canv = benu.benu.Canvas(png, self.w, self.h)

        self.wfmf.render(canv, self.panels['wide'], desc)
        self.zfmf.render(canv, self.panels['zoom'], desc)
        self.plotttm.render(canv, self.panels['plot'], desc)

        canv.save()
        self.i += 1

        return png

def get_progress_bar(name, maxval):
    widgets = ["%s: " % name, progressbar.Percentage(),
               progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,maxval=maxval).start()
    return pbar

def build_framedesc_list(pool_df, wt, zt):
    frames = []

    for z in zt:
        z_frameoffset = np.where(zt == z)[0][0]
        z_frameno = int(z)
        assert z_frameno == z

        row = None

        for idx,hmatchingrow in (pool_df[pool_df['h_framenumber'] == z_frameno]).iterrows():

            if hmatchingrow.isnull().values.any():
                #missing data
                continue

            try:
                t_frameno = int(hmatchingrow['t_framenumber'])
                assert t_frameno == hmatchingrow['t_framenumber']
            except ValueError:
                #nan
                continue

            try:
                w_frameoffset = np.where(wt == t_frameno)[0][0]
                row = hmatchingrow
                w_frameno = t_frameno
            except IndexError:
                continue

        if row is not None:

            w_frame = FMFFrame(w_frameoffset, w_frameno)
            z_frame = FMFFrame(z_frameoffset, z_frameno)
            epoch = idx.asm8.astype(np.int64) / 1e9

            fd = FrameDescriptor(w_frame, z_frame, hmatchingrow, epoch)

            frames.append(fd)

    return frames

if __name__ == "__main__":
    from pprint import pprint

    ZOOM_FMF = '/mnt/strawscience/data/FlyMAD/new_movies/z_new_movie_aversion_h20131030_180450.fmf'
    WIDE_FMF = '/mnt/strawscience/data/FlyMAD/new_movies/w_new_movie_aversion_h20131030_180453.fmf'
    BAG_FILE = '/mnt/strawscience/data/FlyMAD/new_movies/2013-10-30-18-04-55.bag'

    wfmf = FMFTrajectoryPlotter(WIDE_FMF)
    zfmf = FMFTTLPlotter(ZOOM_FMF)

    arena = madplot.Arena()

    print "loading data"
    df = madplot.load_bagfile_single_dataframe(BAG_FILE, arena)
    wt = wfmf.fmf.get_all_timestamps()
    zt = zfmf.fmf.get_all_timestamps()

    frames = build_framedesc_list(df, wt, zt)
    if not frames:
        print "no frames to render"
        sys.exit(0)
    else:
        print len(frames),"frames to render"

    moviet0 = frames[0].epoch
    movielen = (frames[-1].epoch - moviet0)
    movieifi = movielen/len(frames)

    ttlplotter = TTLPlotter(moviet0,movieifi)
    wfmf.t0 = moviet0
    zfmf.t0 = moviet0

    panels = {}
    #left half of screen
    panels["wide"] = wfmf.get_benu_panel(
            device_x0=0, device_x1=0.5*TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )
    panels["zoom"] = zfmf.get_benu_panel(
            device_x0=0.5*TARGET_OUT_W, device_x1=TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )

    actual_w, actual_h = benu.utils.negotiate_panel_size_same_height(panels, TARGET_OUT_W)

    #now add the plot at the bottom
    PH = 200
    PW = actual_w

    actual_h += PH

    panels["plot"] = dict(
            width=PW,
            height=PH,
            dw=PW,dh=PH,
            device_x0=0, device_x1=actual_w,
            device_y0=actual_h-PH, device_y1=actual_h
    )

    moviemaker = MovieMaker(obj_id=os.path.basename(BAG_FILE))

    ass = Assembler(actual_w, actual_h,
                    panels, 
                    wfmf,zfmf,ttlplotter,
                    moviemaker,
    )

    pbar = get_progress_bar(moviemaker.movie_fname, len(frames))

    for i,desc in enumerate(frames):
        ass.render_frame(desc)
        pbar.update(i)

    pbar.finish()

    moviefname = moviemaker.render(os.path.dirname(BAG_FILE))
    print "wrote", moviefname

    

