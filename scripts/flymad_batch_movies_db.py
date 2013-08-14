#!/usr/bin/env python

import time
import os.path
import glob
import tempfile
import shutil

import flymad_compositor
import flymad_moviemaker

def make_movie(wide,zoom,bag,imagepath,filename):
    flymad_compositor.doit(wide,zoom,bag,imagepath=imagepath)
    return flymad_moviemaker.doit(imagepath,finalmov=filename)

if __name__ == "__main__":
    BAG_DATE_FMT = "rosbagOut_%Y-%m-%d-%H-%M-%S.bag"
    FMF_DATE_FMT = "%Y%m%d_%H%M%S.fmf"

    destdir = os.path.expanduser("~/movies")
    inputfmf = os.path.expanduser("/media/DBATH_1TB/please_make_mp4s") 
    print inputfmf
    inputbags = os.path.expanduser("~/flymad_rosbag/")

    fmfs = glob.glob(inputfmf+"/*.fmf")
    bags = glob.glob(inputbags+"/*.bag")
    
    print fmfs
    print bags

    for bag in bags:
        #print 'processing bag file', bag
        btime = time.strptime(os.path.basename(bag), BAG_DATE_FMT)
        matching = {}
        for fmf in fmfs:
            print 'processing fmf', fmf
            try:
                fmffn = os.path.basename(fmf)
                genotype,camn,datestr = fmffn.split("_",2)
            except ValueError:
                print "invalid fmfname", fmf
                continue

            try:
                fmftime = time.strptime(datestr, FMF_DATE_FMT)
            except ValueError:
                print "invalid fmffname", fmf
                continue

            dt = abs(time.mktime(fmftime) - time.mktime(btime))

            if (dt < 20):
                matching[camn] = fmf
                matching["genotype"] = genotype
                matching["date"] = btime

        if matching:
            tmpdir = tempfile.mkdtemp()
            destfn = os.path.join(destdir,
                            "%s_%s.mp4" % (matching["genotype"], 
                                           time.strftime("%Y%m%d_%H%M%S",matching["date"]))
            )

            print "making",destfn
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


