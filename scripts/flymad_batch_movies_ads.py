#!/usr/bin/env python

import time
import os.path
import glob
import tempfile
import shutil
import sys, re

import flymad_compositor
import flymad_moviemaker

from .analysis.th_experiments import DOROTHEA_NAME_RE_BASE
DOROTHEA_NAME_REGEXP = re.compile(r'^' + DOROTHEA_NAME_RE_BASE + '$')

if __name__ == "__main__":
    BAG_DATE_FMT = "%Y-%m-%d-%H-%M-%S.bag"
    FMF_DATE_FMT = "%Y%m%d_%H%M%S"

    destdir = os.path.expanduser("/mnt/strawscience/data/FlyMAD/revision_dorothea/18_02/analysis")
    inputfmf = os.path.expanduser("/mnt/strawscience/data/FlyMAD/revision_dorothea/18_02")
    print inputfmf
    inputbags = os.path.expanduser("/mnt/strawscience/data/FlyMAD/revision_dorothea/18_02/TH_Gal4_bagfiles")

    fmfs = glob.glob(inputfmf+"/*.fmf")
    bags = glob.glob(inputbags+"/*.bag")

    #print fmfs
    #print bags

    for bag in bags:
        #print 'processing bag file', bag
        btime = time.strptime(os.path.basename(bag), BAG_DATE_FMT)
        matching = {}
        for fmf in fmfs:
            fmffn = os.path.basename(fmf)
            matchobj = DOROTHEA_NAME_REGEXP.match(fmffn)
            print 'fmffn',fmffn
            if matchobj is None:
                print '-'*80
                print '-'*80
                print '-'*80
                print 'WARNING: COULD NOT PARSE FILENAME FOR %r'%fmffn
                print '-'*80
                print '-'*80
                print '-'*80
                continue
            parsed_data = matchobj.groupdict()
            print '%s -> %s'%(fmffn,parsed_data)

            try:
                fmftime = time.strptime(parsed_data['datetime'], FMF_DATE_FMT)
            except ValueError:
                print "invalid fmffname", fmf
                continue

            dt = abs(time.mktime(fmftime) - time.mktime(btime))

            if (dt < 20):
                matching["condition"] = parsed_data['condition']
                matching["date"] = btime

        if matching:
            tmpdir = tempfile.mkdtemp()
            destfn = os.path.join(destdir,
                            "%s_%s.mp4" % (matching["condition"],
                                           time.strftime("%Y%m%d_%H%M%S",matching["date"]))
            )

            print "making",destfn
            flymad_moviemaker.doit(imagepath,finalmov=filename)
            #try:
            ok = make_movie(
                    matching["wide"],
                    matching["zoom"],
                    bag,
                    tmpdir,
                    destfn
            )
            """"except:
                print "error making movie:", bag
                pass
"""
            shutil.rmtree(tmpdir)


