#!/usr/bin/env python
import sys
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

import roslib; roslib.load_manifest('flymad')
import rosbag
import flymad.madplot as madplot
from flymad.th_experiments import DOROTHEA_NAME_RE_BASE, DOROTHEA_BAGDIR, DOROTHEA_MP4DIR

DOROTHEA_NAME_REGEXP = re.compile(r'^' + DOROTHEA_NAME_RE_BASE + '$')

USE_MULTIPROCESSING = True

Pair = collections.namedtuple('Pair', 'fmf bag maxt')

FMF_DATE_FMT = "%Y%m%d_%H%M%S"

assert benu.__version__ >= "0.1.0"

TARGET_OUT_W, TARGET_OUT_H = 1280, 1024
MARGIN = 2
TZNAME = 'CET'

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

        self.wfmf.render(canv, self.panels['wide'], desc)
        self.zfmf.render(canv, self.panels['zoom'], desc)

        canv.save()
        self.i += 1

        return png

def doit_using_framenumber(user_data):
    match, mp4_dir, show_theta, show_velocity, pre_frames, post_frames = user_data

    zoomf = match.fmf
    rosbagf = match.bag
    maxt = match.maxt

    moviemaker = madplot.MovieMaker(basename=os.path.basename(zoomf), fps=15)
    target_moviefname = moviemaker.get_target_movie_name(mp4_dir)
    if os.path.exists(target_moviefname):
        print 'target %r exists: skipping movie'%(target_moviefname,)
        moviemaker.cleanup()
        return

    arena = madplot.Arena(False)
    zoom = madplot.FMFImagePlotter(zoomf, 'z_frame')
    zoom.enable_color_correction(brightness=15, contrast=1.5)
    wide = madplot.ArenaPlotter(arena,tzname=TZNAME)

    wide.show_theta = show_theta
    wide.show_velocity = show_velocity
    wide.show_epoch = True
    wide.show_framenumber = True
    wide.show_lxly = True
    wide.show_fxfy = True

    renderlist = []

    zoom_ts = zoom.fmf.get_all_timestamps().tolist()
    df = madplot.load_bagfile_single_dataframe(rosbagf, arena, ffill=True,
                                               tzname=TZNAME)
    t0 = df.index[0].asm8.astype(np.int64) / 1e9

    # find start and stop frames based on laser transition ----
    laser_power = df['laser_power'].values
    framenumber = df['h_framenumber'].values

    laser_power = np.array( laser_power, copy=True )
    laser_power[ np.isnan(laser_power) ] = 0

    dlaser = laser_power[1:]-laser_power[:-1]
    laser_transitions = np.nonzero(dlaser)[0]
    startframenumber = -np.inf
    stopframenumber = np.inf
    if pre_frames is None:
        assert post_frames is None
    if pre_frames is not None:
        assert post_frames is not None

        for i,idx in enumerate(laser_transitions):
            if i==0:
                startframenumber = framenumber[idx-pre_frames]
            elif i==1:
                stopframenumber = framenumber[idx+post_frames]
            else:
                print 'ERROR: unknown laser transition in %r'%zoomf
                moviemaker.cleanup()
                return

    # got start and stop frames -----

    for idx,group in df.groupby('h_framenumber'):
        # limit movies to only frames we want
        this_framenumber = group['h_framenumber'].values[0]
        if not (startframenumber <= this_framenumber and this_framenumber <= stopframenumber):
            continue

        #did we save a frame ?
        try:
            frameoffset = zoom_ts.index(idx)
        except ValueError:
            #missing frame (probbably because the video was not recorded at
            #full frame rate
            continue

        frame = madplot.FMFFrame(offset=frameoffset, timestamp=idx)
        row = group.dropna(subset=['tobj_id']).tail(1)
        if len(row):
            if maxt > 0:
                dt = (row.index[0].asm8.astype(np.int64) / 1e9) - t0
                if dt > maxt:
                    break

            desc = madplot.FrameDescriptor(
                                None,
                                frame,
                                row,
                                row.index[0].asm8.astype(np.int64) / 1e9)

            renderlist.append(desc)

    if len(renderlist)==0:
        moviemaker.cleanup()
        return

    wide.t0 = t0

    panels = {}
    #left half of screen
    panels["wide"] = wide.get_benu_panel(
            device_x0=0, device_x1=0.5*TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )
    panels["zoom"] = zoom.get_benu_panel(
            device_x0=0.5*TARGET_OUT_W, device_x1=TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )

    actual_w, actual_h = benu.utils.negotiate_panel_size_same_height(panels, TARGET_OUT_W)

    ass = Assembler(actual_w, actual_h,
                    panels,
                    wide,zoom,
                    moviemaker,
    )


    if not USE_MULTIPROCESSING:
        pbar = madplot.get_progress_bar(moviemaker.movie_fname, len(renderlist))

    for i,desc in enumerate(renderlist):
        ass.render_frame(desc)
        if not USE_MULTIPROCESSING:
            pbar.update(i)

    if not USE_MULTIPROCESSING:
        pbar.finish()

    if not os.path.exists(mp4_dir):
        os.makedirs(mp4_dir)

    moviefname = moviemaker.render(mp4_dir)
    print "wrote", moviefname

    moviemaker.cleanup()

def get_matching_bag(fmftime, bagdir):
    bags = []
    for bag in glob.glob(os.path.join(bagdir,'*.bag')):
        btime = madplot.strptime_bagfile(bag)
        try:
            dt = abs(time.mktime(fmftime) - time.mktime(btime))
        except:
            print 'fmftime',fmftime
            print 'btime',btime
            raise
        bags.append( (bag, dt) )

    #sort based on dt (smallest dt first)
    bags.sort(key=operator.itemgetter(1))
    bag,dt = bags[0]

    if dt < 10:
        return bag
    else:
        return None

def get_matching_fmf_and_bag(fmf_dir, maxtime,
                             bagdir):

    matching = []
    if not os.path.isdir(bagdir):
        raise RuntimeError('bagdir not a directory')

    for fmffile in glob.glob(os.path.join(fmf_dir,'*.fmf')):
        fmfname = os.path.basename(fmffile)
        matchobj = DOROTHEA_NAME_REGEXP.match(fmfname)

        if matchobj is None:
            print "error: incorrectly named fmf file?", fmffile
            print 'fmfname',fmfname
            continue
        parsed_data = matchobj.groupdict()
        #print '%s -> %s'%(fmfname,parsed_data)

        fmftime = time.strptime(parsed_data['datetime'], FMF_DATE_FMT)
        bagfile = get_matching_bag(fmftime, bagdir)
        if bagfile is None:
            print "no bag for",fmffile
        else:
                matching.append( Pair(fmf=fmffile, bag=bagfile, maxt=maxtime) )

    return matching

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fmfdir', nargs=1, help='path to data (a dir of fmfs and subdir of bags)')
#    parser.add_argument('--genotype', required=True, help='genotype (the prefix of the fmfs; cs, Moonw, etc)')
    parser.add_argument('--disable-multiprocessing', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--show-theta', action='store_true', default=False)
    parser.add_argument('--show-velocity', action='store_true', default=False)
    parser.add_argument('--max-time', type=int, default=0, help='max time of video')
    parser.add_argument('--outdir', default=None, help='dir to save mp4s')
    parser.add_argument('--bagdir', default=None, help='dir to save mp4s')
    parser.add_argument('--n-frames-pre-laser', type=int, default=None)
    parser.add_argument('--n-frames-post-laser', type=int, default=None)

    args = parser.parse_args()
    fmfdir = args.fmfdir[0]

    if not os.path.isdir(fmfdir):
        parser.error('fmfdir must be a directory')

    if args.bagdir is None:
        args.bagdir = DOROTHEA_BAGDIR

    if args.outdir is None:
        args.outdir = DOROTHEA_MP4DIR

    matching = [(m,args.outdir,args.show_theta,args.show_velocity,
                 args.n_frames_pre_laser, args.n_frames_post_laser,
                 )\
                for m in get_matching_fmf_and_bag(fmfdir, args.max_time,
                                                  args.bagdir,
                                                  )]
    print len(matching),"matching"
    if len(matching)==0:
        sys.exit(0)

    if args.dry_run:
        for match in matching:
            print match[0]
        sys.exit(0)

    if (not args.disable_multiprocessing) and USE_MULTIPROCESSING:
        print "using multiprocessing"
        pool = multiprocessing.Pool()
        pool.map(doit_using_framenumber, matching)
        pool.close()
        pool.join()
    else:
        for match in matching:
            doit_using_framenumber(match)
