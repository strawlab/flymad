#!/usr/bin/env python

import os.path
import sh

def doit(idir, fps=30, tmpmov=None, finalmov=None):
    fps = 30

    if tmpmov is None:
        tmpmov = "%s/movie.y4m" % idir

    sh.mplayer("mf://%s/*.png" % idir,
               "-mf", "fps=%d" % fps,
               "-vo", "yuv4mpeg:file=%s" % tmpmov,
               "-ao", "null",
               "-nosound", "-noframedrop", "-benchmark", "-nolirc"
    )

    if finalmov is None:
        finalmov = "%s/movie.mp4" % idir

    sh.x264("--output=%s" % finalmov,
            "%s" % tmpmov,
    )

    try:
        os.unlink(tmpmov)
    except OSError:
        pass

    return finalmov

if __name__ == "__main__":
    doit("./")
