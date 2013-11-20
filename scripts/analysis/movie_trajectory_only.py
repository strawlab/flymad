#!/usr/bin/env python

import time
import os.path
import glob
import tempfile
import shutil
import re
import collections
import operator
import multiprocessing

import motmot.FlyMovieFormat.FlyMovieFormat as fmf
import pandas as pd
import numpy as np

import benu.benu
import benu.utils

import roslib; roslib.load_manifest('rosbag')
import rosbag

import madplot

class Assembler:
    def __init__(self, w, h, panels, wfmf, moviemaker):
        self.panels = panels
        self.w = w
        self.h = h
        self.wfmf = wfmf
        self.i = 0
        self.moviemaker = moviemaker

    def render_frame(self, desc):
        assert isinstance(desc, madplot.FrameDescriptor)

        png = self.moviemaker.next_frame()
        canv = benu.benu.Canvas(png, self.w, self.h)

        try:
            self.wfmf.render(canv, self.panels['wide'], desc)
            canv.save()
            self.i += 1
        except Exception:
            import traceback
            traceback.print_exc()
            print "error rendering", desc.df

        return png


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag file')
    parser.add_argument('--wide-fmf', help='wide fmf file to render the trajectory over')
    parser.add_argument('--outdir', help='destination directory for mp4')
    parser.add_argument('--show-laser', action='store_true', default=False, help='show laser postion')
    parser.add_argument('--show-target', action='store_true', default=False, help='show targeted fly')
    parser.add_argument('--show-timestamp', action='store_true', default=False, help='show frame timestamp (likely framenumber)')
    parser.add_argument('--show-epoch', action='store_true', default=False, help='show epoch')
    parser.add_argument('--fps', type=int, default=20, help='framenumber')

    args = parser.parse_args()
    path = args.path[0]

    arena = madplot.Arena(False)
    df = madplot.load_bagfile_single_dataframe(path, arena, ffill=False)

    objids = df['tobj_id'].dropna().unique()

    wfmf = madplot.FMFMultiTrajectoryPlotter(args.wide_fmf, objids, maxlen=400)
    wfmf.show_lxly = args.show_laser
    wfmf.show_fxfy = args.show_target
    wfmf.show_timestamp = args.show_timestamp
    wfmf.show_epoch = args.show_epoch

    fmfwidth = wfmf.width
    fmfheight = wfmf.height

    try:
        print 'loading w timestamps'
        wts = wfmf.fmf.get_all_timestamps()
    except AttributeError:
        wts = df['t_framenumber'].dropna().unique()

    pbar = madplot.get_progress_bar("computing frames", len(wts))

    frames = []
    for i,(wt0,wt1) in enumerate(madplot.pairwise(wts)):

        pbar.update(i)

        fdf = madplot.get_framedf(df,wt1)
        lfdf = len(fdf)

        if lfdf:
            try:
                most_recent_time = madplot.get_framedf(df, wt1).index[-1]

                minidf = df[madplot.get_framedf(df, wt0).index[0]:most_recent_time]
                last_head_row = minidf[minidf['h_framenumber'].notnull()].tail(1)

                last_laser_row = minidf[minidf['lobj_id'].notnull()].tail(1)

                w_frame = madplot.FMFFrame(
                            *madplot.get_offset_and_nearest_fmf_timestamp(wts, float(wt1)))

                epoch = most_recent_time.asm8.astype(np.int64) / 1e9
                fd = madplot.FrameDescriptor(
                                w_frame, None, 
                                fdf.merge(pd.merge(last_head_row, last_laser_row, 'outer'),'outer'),
                                epoch
                )

                frames.append(fd)

                #FIXME: fails at movies longer than 27000 frames for unknown
                #reasons
                if len(frames) > 27000: break

                if 'TEST_MOVIES' in os.environ:
                    if len(frames) > 50: break

            except (IndexError, TypeError):
                import traceback
                traceback.print_exc()
                print "frame",i,"error: no frame descriptor generated\n",fdf

    pbar.finish()

    if not frames:
        print "no frames to render"
        sys.exit(0)
    else:
        print len(frames),"frames to render"

    moviet0 = frames[0].epoch
    movielen = (frames[-1].epoch - moviet0)
    movieifi = movielen/len(frames)
    wfmf.t0 = moviet0

    panels = {}
    #left half of screen
    panels["wide"] = wfmf.get_benu_panel(
            device_x0=0, device_x1=fmfwidth,
            device_y0=0, device_y1=fmfheight
    )

    actual_w, actual_h = benu.utils.negotiate_panel_size_same_height(panels, fmfwidth)

    moviemaker = madplot.MovieMaker(obj_id=os.path.basename(path), fps=args.fps)

    ass = Assembler(actual_w, actual_h,
                    panels, 
                    wfmf,
                    moviemaker,
    )

    pbar = madplot.get_progress_bar(moviemaker.movie_fname, len(frames))

    for i,desc in enumerate(frames):
        ass.render_frame(desc)
        pbar.update(i)

    pbar.finish()

    moviefname = moviemaker.render(args.outdir if args.outdir else os.path.dirname(path))
    print "wrote", moviefname

    moviemaker.cleanup()

