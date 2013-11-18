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

import roslib; roslib.load_manifest('rosbag')
import rosbag

import madplot

USE_MULTIPROCESSING = True

Pair = collections.namedtuple('Pair', 'fmf bag maxt')

FMF_DATE_FMT = "%Y%m%d_%H%M%S.fmf"

assert benu.__version__ >= "0.1.0"

TARGET_OUT_W, TARGET_OUT_H = 1280, 1024
MARGIN = 2

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

def doit_using_framenumber(match):
    zoomf = match.fmf
    rosbagf = match.bag
    maxt = match.maxt

    arena = madplot.Arena()
    zoom = madplot.FMFImagePlotter(zoomf, 'z_frame')
    zoom.enable_color_correction(brightness=15, contrast=1.5)
    wide = madplot.ArenaPlotter(arena)

    renderlist = []

    zoom_ts = zoom.fmf.get_all_timestamps().tolist()
    df = madplot.load_bagfile_single_dataframe(rosbagf, arena, ffill=True)
    t0 = df.index[0].asm8.astype(np.int64) / 1e9

    for idx,group in df.groupby('h_framenumber'):
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

    moviemaker = madplot.MovieMaker(obj_id=os.path.basename(zoomf), fps=14)

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

    mdir = 'mp4s'
    if not os.path.exists(mdir):
        os.makedirs(mdir)

    moviefname = moviemaker.render(mdir)
    print "wrote", moviefname

    moviemaker.cleanup()
    

def make_movie(wide,zoom,bag,imagepath,filename):
    flymad_compositor.doit(wide,zoom,bag,imagepath=imagepath)
    return flymad_moviemaker.doit(imagepath,finalmov=filename)

def get_matching_bag(fmftime, bagdir):
    bags = []
    for bag in glob.glob(os.path.join(bagdir,'*.bag')):
        btime = madplot.strptime_bagfile(bag)
        dt = abs(time.mktime(fmftime) - time.mktime(btime))
        bags.append( (bag, dt) )

    #sort based on dt (smallest dt first)    
    bags.sort(key=operator.itemgetter(1))
    bag,dt = bags[0]

    if dt < 10:
        return bag
    else:
        return None

def get_bag_re(gt):
    #the non-control MW experiment movies are names Moonw_movie
    #return re.compile("%s_movie_([abh+]{1,3})_([0-9_]{1,3})(2013)(.*)" % gt)
    return re.compile("%s_([abh+]{1,3})_([0-9_]+)(2013)(.*)" % gt)



def get_matching_fmf_and_bag(gt, base_dir, maxtime=0):

    bag_re = get_bag_re(gt)
    matching = []

    for fmffile in glob.glob(os.path.join(base_dir,'%s_*.fmf' % gt)):
        fmfname = os.path.basename(fmffile)
        try:
            target,trial,year,date = bag_re.search(fmfname).groups()
            bagdir = os.path.join(base_dir,'%s_%s_%s' % (gt, target, trial))
            if os.path.isdir(bagdir):
                #we found a directory with matching bag files
                fmftime = time.strptime("%s%s" % (year,date), FMF_DATE_FMT)
                bagfile = get_matching_bag(fmftime, bagdir)
                if bagfile is None:
                    print "no bag for",fmffile
                else:
                    matching.append( Pair(fmf=fmffile, bag=bagfile, maxt=maxtime) )
            else:
                print "no bags for",fmffile
            #    
        except AttributeError:
            #no regex match
            print "error: incorrectly named fmf file?", fmffile

    return matching

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to data (a dir of fmfs and subdir of bags)')
    parser.add_argument('--genotype', required=True, help='genotype (the prefix of the fmfs; cs, Moonw, etc)')
    parser.add_argument('--disable-multiprocessing', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--max-time', type=int, default=0, help='max time of video')

    args = parser.parse_args()
    path = args.path[0]

    if not os.path.isdir(path):
        parser.error('must be a directory')

    matching = get_matching_fmf_and_bag(args.genotype, path, args.max_time)
    print len(matching),"matching"

    if args.dry_run:
        for match in matching:
            print match
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


