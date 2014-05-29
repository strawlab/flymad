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
import sys

import motmot.FlyMovieFormat.FlyMovieFormat as fmf
import pandas as pd
import numpy as np

import benu.benu
import benu.utils

import roslib; roslib.load_manifest('flymad')
import flymad.madplot as madplot

Cut = collections.namedtuple('Cut', 'start end dest')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to data (a dir of bags and one fmf)')

    args = parser.parse_args()
    path = args.path[0]

    if not os.path.isdir(path):
        parser.error('must be a directory')

    bags = glob.glob(os.path.join(path, "*.bag"))
    fmffile = fmf.FlyMovie(glob.glob(os.path.join(path, "*.fmf"))[0])

    ts = fmffile.get_all_timestamps().tolist()

    arena = madplot.Arena(False)

    cuts = []
    for bag in sorted(bags):
        geom, dfs = madplot.load_bagfile(bag, arena)
        h_df = dfs["ttm"]

        fn = h_df['h_framenumber']
        start = fn.ix[0]
        stop = fn.ix[-1]

        dn = os.path.dirname(bag)
        btime = madplot.strptime_bagfile(bag)
        destfmfname = "%s%d%02d%02d_%02d%02d%02d.fmf" % (os.path.basename(dn),
                        btime.tm_year,btime.tm_mon,btime.tm_mday,
                        btime.tm_hour,btime.tm_min,btime.tm_sec)

        destfmf = os.path.join(dn,'..',destfmfname)

        cuts.append( Cut(start,stop,os.path.abspath(destfmf)) )

    #many iterations, yuck. meh.
    while cuts:
        try:
            cut = cuts.pop(0)
        except IndexError:
            continue

        startt = endt = None
        for t in ts:
            if startt is None and t >= cut.start:
                startt = t
            if endt is None and startt is not None and t >= cut.end:
                endt = t
                break

        if endt is not None:
            print "cut", cut
            dest = fmf.FlyMovieSaver(
                            cut.dest,
                            format=fmffile.format,
                            bits_per_pixel=fmffile.bits_per_pixel)
            for i in range(ts.index(startt),ts.index(endt)):
                frame, timestamp = fmffile.get_frame(i)
                dest.add_frame(frame,timestamp)
            dest.close()
        else:
            print "no cut found for",cut.dest


