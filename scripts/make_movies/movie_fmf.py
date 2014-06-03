#!/usr/bin/env python

import sys
import os.path

import numpy as np
import cv2

import roslib; roslib.load_manifest('flymad')
import flymad.madplot as madplot

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to fmf file')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--outdir', help='destination directory for mp4')

    args = parser.parse_args()
    path = args.path[0]

    fmf = madplot.FMFImagePlotter(path, 'none')
    fmf.enable_color_correction(brightness=20, contrast=1.5)

    frames = fmf.fmf.get_all_timestamps()
    moviemaker = madplot.MovieMaker(basename=os.path.basename(path)+'_plain', fps=args.fps)
    pbar = madplot.get_progress_bar(moviemaker.movie_fname, len(frames))

    for i,ts in enumerate(frames):
        f = fmf.get_frame_number(i)
        png = moviemaker.next_frame()
        cv2.imwrite(png, f)
        pbar.update(i)

        if 'TEST_MOVIES' in os.environ:
            if i > 50: break

    pbar.finish()

    moviefname = moviemaker.render(args.outdir if args.outdir else os.path.dirname(path))
    print "wrote", moviefname

    moviemaker.cleanup()

    

