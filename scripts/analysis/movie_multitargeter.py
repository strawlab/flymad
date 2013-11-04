import sys
import os.path

import pandas as pd
import numpy as np
import progressbar

import benu.benu
import benu.utils

import madplot

assert benu.__version__ >= "0.1.0"

TARGET_OUT_W, TARGET_OUT_H = 1280, 1024
MARGIN = 2

import madplot
import motmot.FlyMovieFormat.FlyMovieFormat

class Assembler:
    def __init__(self, w, h, panels, wfmf, zfmf, moviemaker):
        self.panels = panels
        self.w = w
        self.h = h
        self.wfmf = wfmf
        self.zfmf = zfmf
        self.i = 0
        self.moviemaker = moviemaker

    def render_frame(self, desc):
        assert isinstance(desc, madplot.FrameDescriptor)

        png = self.moviemaker.next_frame()
        canv = benu.benu.Canvas(png, self.w, self.h)

        try:
            self.wfmf.render(canv, self.panels['wide'], desc)
            self.zfmf.render(canv, self.panels['zoom'], desc)

            canv.save()
            self.i += 1
        except Exception:
            import traceback
            traceback.print_exc()
            print "error rendering", desc.df

        return png

def get_offset_and_nearest_fmf_timestamp(tss, timestamp):
    at_or_before_timestamp_cond = tss <= timestamp
    nz = np.nonzero(at_or_before_timestamp_cond)[0]
    if len(nz)==0:
        raise ValueError("no frames at or before timestamp given")
    return nz[-1], tss[nz[-1]]

def get_framedf(df, framenumber):
    return df[df['t_framenumber'] == framenumber]

def iter_last_and_this(j):
    for i,_j in enumerate(j):
        if i == 0:
            last = _j
            continue
        yield last, _j
        last = _j

if __name__ == "__main__":

    WIDE_FMF = '/mnt/strawscience/data/FlyMAD/new_movies/reiser/reiser_movie_120131030_173444.fmf'
    ZOOM_FMF = '/mnt/strawscience/data/FlyMAD/new_movies/reiser/reiser_movie_220131030_173442.fmf'
    BAG_FILE = '/mnt/strawscience/data/FlyMAD/new_movies/reiser/2013-10-30-17-34-49.bag'

    arena = madplot.Arena()
    df = madplot.load_bagfile_single_dataframe(BAG_FILE, arena, ffill=False)

    wfmf = madplot.FMFMultiTrajectoryPlotter(WIDE_FMF)
    zfmf = madplot.FMFTTLPlotter(ZOOM_FMF)

    wts = wfmf.fmf.get_all_timestamps()
    zts = zfmf.fmf.get_all_timestamps()

    frames = []    

    for wt0,wt1 in iter_last_and_this(wts):

        fdf = get_framedf(df,wt1)
        lfdf = len(fdf)

        print wt1,lfdf

        if lfdf:
            try:
                most_recent_time = get_framedf(df, wt1).index[-1]

                minidf = df[get_framedf(df, wt0).index[0]:most_recent_time]
                last_head_row = minidf[minidf['h_framenumber'].notnull()].tail(1)

                last_laser_row = minidf[minidf['lobj_id'].notnull()].tail(1)

                z_frame = madplot.FMFFrame(
                            *get_offset_and_nearest_fmf_timestamp(zts, float(last_head_row['h_framenumber'])))
                w_frame = madplot.FMFFrame(
                            *get_offset_and_nearest_fmf_timestamp(wts, float(wt1)))

                epoch = most_recent_time.asm8.astype(np.int64) / 1e9
                fd = madplot.FrameDescriptor(
                                w_frame, z_frame, 
                                fdf.merge(pd.merge(last_head_row, last_laser_row, 'outer'),'outer'),
                                epoch
                )

                frames.append(fd)

            except (IndexError, TypeError):
                import traceback
                traceback.print_exc()
                print "error preparing frame descriptor for\n",fdf

    if not frames:
        print "no frames to render"
        sys.exit(0)
    else:
        print len(frames),"frames to render"

    moviet0 = frames[0].epoch
    movielen = (frames[-1].epoch - moviet0)
    movieifi = movielen/len(frames)

    ttlplotter = madplot.TTLPlotter(moviet0,movieifi)
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

    moviemaker = madplot.MovieMaker(obj_id=os.path.basename(BAG_FILE))

    ass = Assembler(actual_w, actual_h,
                    panels, 
                    wfmf,zfmf,
                    moviemaker,
    )

    pbar = madplot.get_progress_bar(moviemaker.movie_fname, len(frames))

    for i,desc in enumerate(frames):
        ass.render_frame(desc)
        pbar.update(i)

    pbar.finish()

    moviefname = moviemaker.render(os.path.dirname(BAG_FILE))
    print "wrote", moviefname

