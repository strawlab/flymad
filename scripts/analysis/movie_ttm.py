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

class _FMFPlotter:

    def __init__(self, path):
        self.fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(path)
        _,self.f0 = self.fmf.get_frame(0)

    def get_frame(self, timestamp_as_framenumber, color=False):
        rel_framenumber = timestamp_as_framenumber - self.f0
        f,ts = self.fmf.get_frame(rel_framenumber)

        assert ts == timestamp_as_framenumber

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

    def get_frame0_framenumber(self):
        return self.f0

class FMFTrajectoryPlotter(_FMFPlotter):

    name = 'w'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.xhist = collections.deque(maxlen=MAXLEN)
        self.yhist = collections.deque(maxlen=MAXLEN)

    def render(self, canv, panel, framenumber, row):
        x,y = row['fly_x'],row['fly_y']

        self.xhist.append(x)
        self.yhist.append(y)

        img = self.get_frame(framenumber)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )
            canv.scatter( [x],
                          [y],
                          color_rgba=(0,1,0,0.3), radius=2.0 )
            canv.scatter( self.xhist,
                          self.yhist,
                          color_rgba=(0,1,0,0.3), radius=0.5 )



class FMFTTLPlotter(_FMFPlotter):

    name = 'f'

    def __init__(self, path):
        _FMFPlotter.__init__(self, path)
        self.hx = self.hy = 0

    def render(self, canv, panel, framenumber, row):
        hx,hy = row['head_x'],row['head_y']
        hdx,hdy = row['head_dx'],row['head_dy']
        bdx,bdy = row['body_dx'],row['body_dy']

        img = self.get_frame(framenumber)
        with canv.set_user_coords_from_panel(panel):
            canv.imshow(img, 0,0, filter='best' )
            canv.scatter( [hx],
                          [hy],
                          color_rgba=(0,1,0,0.3), radius=10.0 )

#            canv.scatter( [hx-hdx],
#                          [hy-hdy],
#                          color_rgba=(0,0,1,0.3), radius=5.0 )

#            canv.scatter( [hx-bdx],
#                          [hy-bdy],
#                          color_rgba=(1,0,0,0.3), radius=5.0 )


class TTLPlotter:

    IFI  = 1.0/100
    HIST = 3.0

    def __init__(self):
        maxlen = self.HIST / self.IFI
        self.dx = collections.deque(maxlen=maxlen)
        self.dy = collections.deque(maxlen=maxlen)

    def render(self, canv, panel, framenumber, row):
        head_dx, head_dy = row['head_dx'], row['head_dy']

        self.dx.appendleft(head_dx)
        self.dy.appendleft(head_dy)

        time = np.array(range(len(self.dx)))*self.IFI*-1.0

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
        self.tmpdir = "/tmp/tmp4e2EPu2013-10-27-16-48-35.bag"#tempfile.mkdtemp(str(obj_id), dir=tmpdir)
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

        self.w0 = self.wfmf.get_frame0_framenumber()
        self.z0 = self.zfmf.get_frame0_framenumber()

        self.i = 0
        self.moviemaker = moviemaker

    def render_row(self, row):
        if row.isnull().values.any():
            print "row missing data"
            return None

        w_framenumber = row['t_framenumber']
        z_framenumber = row['h_framenumber']

        if w_framenumber < self.w0:
            print "no frame for wide fmf"
            return None
        if z_framenumber < self.z0:
            print "no frame for zoom fmf"
            return None

        png = self.moviemaker.next_frame()
        canv = benu.benu.Canvas(png, self.w, self.h)

        self.wfmf.render(canv, self.panels['wide'], w_framenumber, row)
        self.zfmf.render(canv, self.panels['zoom'], z_framenumber, row)
        self.plotttm.render(canv, self.panels['plot'], self.i, row)

        canv.save()
        self.i += 1

        return png

def get_progress_bar(name, maxval):
    widgets = ["%s: " % name, progressbar.Percentage(),
               progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,maxval=maxval).start()
    return pbar

if __name__ == "__main__":
    png = "test.png"

    from pprint import pprint

    wfmf = FMFTrajectoryPlotter('/mnt/strawscience/data/FlyMAD/aversion_movies/3/w_aversion_movie_new2_a20131027_164830.fmf')
    zfmf = FMFTTLPlotter('/mnt/strawscience/data/FlyMAD/aversion_movies/3/z_aversion_movie_new2_a20131027_164832.fmf')
    b = '/mnt/strawscience/data/FlyMAD/aversion_movies/3/2013-10-27-16-48-35.bag'

    arena = madplot.Arena()
    pool_df = madplot.load_bagfile_single_dataframe(b, arena)
    #resample to 10ms to better match the fmf videos (100 fps)
    pool_df = pool_df.resample('10L', how='last')

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

    moviemaker = MovieMaker(obj_id=os.path.basename(b))

    ass = Assembler(actual_w, actual_h,
                    panels, 
                    wfmf,zfmf,TTLPlotter(),
                    moviemaker,
    )

    pbar = get_progress_bar(moviemaker.movie_fname, len(pool_df))

    for i,(ix,row) in enumerate(pool_df.iterrows()):
        ass.render_row(row)
        pbar.update(i)

    pbar.finish()

    moviemaker.render(os.path.dirname(b))

    

