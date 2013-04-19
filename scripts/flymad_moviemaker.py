import sh

tmpdir = "./"
fps = 30

sh.mplayer("mf://%s*.png" % tmpdir,
           "-mf", "fps=%d" % fps,
           "-vo", "yuv4mpeg:file=%s/movie.y4m" % tmpdir,
           "-ao", "null",
           "-nosound", "-noframedrop", "-benchmark", "-nolirc"
)

sh.x264("--profile=baseline",
        "--output=%smovie.mp4" % tmpdir,
        "%smovie.y4m" % tmpdir,
)
