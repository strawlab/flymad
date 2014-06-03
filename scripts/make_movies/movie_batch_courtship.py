#!/usr/bin/env python

import time
import os.path
import glob
import tempfile
import shutil
import pprint
import itertools
import argparse
import sh

import roslib; roslib.load_manifest('flymad')
import flymad.filename_regexes as filename_regexes

import courtship_compositor

def make_movie(wide,zoom,bag,imagepath,filename,rfps,mp4fps):
    #composite png frames
    courtship_compositor.composite_fmfs(wide,zoom,bag,imagepath,rfps)

    #write x264 mp4
    tmpmov = "%s/movie.y4m" % imagepath

    sh.mplayer("mf://%s/*.png" % imagepath,
               "-mf", "fps=%d" % mp4fps,
               "-vo", "yuv4mpeg:file=%s" % tmpmov,
               "-ao", "null",
               "-nosound", "-noframedrop", "-benchmark", "-nolirc"
    )

    sh.x264("--output=%s" % filename,
            "%s" % tmpmov,
    )

    try:
        os.unlink(tmpmov)
    except OSError:
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bagdir', type=str, required=True,
                        help='directory of bag files')
    parser.add_argument('--fmfdir', type=str, required=True,
                        help='directory of fmf files')
    parser.add_argument('--outdir', type=str, required=True,
                        help='directory to write mp4 files')
    parser.add_argument('--fps', type=int, default=15,
                        help='render this many fps from the input data')
    parser.add_argument('--speed', type=int, default=2,
                        help='render with mp4 at this many times the rendered fps. '\
                             'for example, speed=2 and fps=15 yields a 2x realtime '\
                             '(30fps) mp4')
    parser.add_argument('--dry-run', action='store_true',
                        help='dont write movies')

    args = parser.parse_args()

    rfps = args.fps
    mp4fps = int(args.speed * rfps)

    matches = filename_regexes.get_matching_files(
                    args.fmfdir,"fmf",
                    args.bagdir,"bag")

    #group by bag name. quite unsure why itertools.groupby doesnt work here
    gs = {k.bag:[] for k in matches}
    for m in matches:
        gs[m.bag].append(m)

    #for bagname,group in itertools.groupby(matches, lambda x: x.bag):
    for bagname,group in gs.iteritems():
        try:
            fmfa,fmfb = group
        except ValueError:
            print "ERROR: 2 fmfs not found for", bagname
            continue

        if fmfa.fmf_info['desc'] != fmfb.fmf_info['desc']:
            print "ERROR: genotype not identical for fmfs", fmfa.fmf, fmfb.fmf
            continue

        print "MATCH: %s\n\t%s (dt: %s)\n\t%s (dt: %s)" % (bagname,fmfa.fmf,fmfa.dt,fmfb.fmf,fmfb.dt)
        
        destfn = os.path.join(args.outdir,
                        "%s_%s.mp4" % (fmfa.fmf_info["desc"], 
                                       time.strftime("%Y%m%d_%H%M%S",fmfa.bag_time))
        )

        print "making",destfn
        if args.dry_run:
            continue

        try:
            tmpdir = tempfile.mkdtemp()

            make_movie(fmfa.fmf if fmfa.fmf_info['camn'] == "wide" else fmfb.fmf,
                       fmfb.fmf if fmfb.fmf_info['camn'] == "zoom" else fmfa.fmf,
                       bagname,
                       tmpdir,
                       destfn,
                       rfps,mp4fps
            )

        except:
            print "error making movie:", bagname
        finally:
            shutil.rmtree(tmpdir)

