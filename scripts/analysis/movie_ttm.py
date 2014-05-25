import sys
import os.path

import numpy as np

import benu.benu
import benu.utils

import madplot

assert benu.__version__ >= "0.1.0"

TARGET_OUT_W, TARGET_OUT_H = 1280, 1024
MARGIN = 2

class Assembler:
    def __init__(self, w, h, panels, wfmf, zfmf, plotttm, moviemaker):
        self.panels = panels
        self.w = w
        self.h = h
        self.wfmf = wfmf
        self.zfmf = zfmf
        self.plotttm = plotttm
        self.i = 0
        self.moviemaker = moviemaker

    def render_frame(self, desc):
        assert isinstance(desc, madplot.FrameDescriptor)

        png = self.moviemaker.next_frame()
        canv = benu.benu.Canvas(png, self.w, self.h)

        self.wfmf.render(canv, self.panels['wide'], desc)
        self.zfmf.render(canv, self.panels['zoom'], desc)
        self.plotttm.render(canv, self.panels['plot'], desc)

        canv.save()
        self.i += 1

        return png

def build_framedesc_list(pool_df, wt, zt):
    frames = []

    for z in zt:
        z_frameoffset = np.where(zt == z)[0][0]
        z_frameno = int(z)
        assert z_frameno == z

        row = None

        for idx,hmatchingrow in (pool_df[pool_df['h_framenumber'] == z_frameno]).iterrows():

            isnull = hmatchingrow.isnull()
            if isnull.any():
                if isnull['theta'] and (isnull.values.sum() == 1):
                    #only theta is missing, that is harmless
                    pass
                else:
                    #missing data
                    continue

            try:
                t_frameno = int(hmatchingrow['t_framenumber'])
                assert t_frameno == hmatchingrow['t_framenumber']
            except ValueError:
                #nan
                continue

            try:
                w_frameoffset = np.where(wt == t_frameno)[0][0]
                row = hmatchingrow
                w_frameno = t_frameno
            except IndexError:
                continue

        if row is not None:

            w_frame = madplot.FMFFrame(w_frameoffset, w_frameno)
            z_frame = madplot.FMFFrame(z_frameoffset, z_frameno)
            epoch = idx.asm8.astype(np.int64) / 1e9

            fd = madplot.FrameDescriptor(w_frame, z_frame, hmatchingrow, epoch)

            frames.append(fd)

    return frames

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag file')
    parser.add_argument('--wide-fmf', help='wide fmf file to render the trajectory over', required=True)
    parser.add_argument('--zoom-fmf', help='wide fmf file to render the trajectory over', required=True)
    parser.add_argument('--outdir', help='destination directory for mp4')

    args = parser.parse_args()

    BAG_FILE = args.path[0]
    ZOOM_FMF = args.zoom_fmf
    WIDE_FMF = args.wide_fmf

    wfmf = madplot.FMFTrajectoryPlotter(WIDE_FMF)
    zfmf = madplot.FMFTTLPlotter(ZOOM_FMF)
    zfmf.enable_color_correction(brightness=20, contrast=1.5)

    arena = madplot.Arena(False)

    print "loading data"
    df = madplot.load_bagfile_single_dataframe(BAG_FILE, arena, ffill=True)
    wt = wfmf.fmf.get_all_timestamps()
    zt = zfmf.fmf.get_all_timestamps()

    if len(df['tobj_id'].dropna().unique()) != 1:
        print "TTM movies require single unique object IDs, I think..."
        sys.exit(1)

    frames = build_framedesc_list(df, wt, zt)
    if not frames:
        print "no frames to render"
        sys.exit(0)
    else:
        print len(frames),"frames to render"

    moviet0 = frames[0].epoch
    movielen = (frames[-1].epoch - moviet0)
    movieifi = movielen/len(frames)

    ttlplotter = madplot.TTLPlotter(moviet0,movieifi)
    wfmf.t0 = moviet0
    zfmf.t0 = moviet0

    panels = {}
    #left half of screen
    panels["wide"] = wfmf.get_benu_panel(
            device_x0=0, device_x1=0.5*TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )
    panels["zoom"] = zfmf.get_benu_panel(
            device_x0=0.5*TARGET_OUT_W, device_x1=TARGET_OUT_W,
            device_y0=0, device_y1=TARGET_OUT_H
    )

    actual_w, actual_h = benu.utils.negotiate_panel_size_same_height(panels, TARGET_OUT_W)

    #now add the plot at the bottom
    PH = 200
    PW = actual_w

    actual_h += PH

    panels["plot"] = dict(
            width=PW,
            height=PH,
            dw=PW,dh=PH,
            device_x0=0, device_x1=actual_w,
            device_y0=actual_h-PH, device_y1=actual_h
    )

    moviemaker = madplot.MovieMaker(basename=os.path.basename(BAG_FILE))

    ass = Assembler(actual_w, actual_h,
                    panels, 
                    wfmf,zfmf,ttlplotter,
                    moviemaker,
    )

    pbar = madplot.get_progress_bar(moviemaker.movie_fname, len(frames))

    for i,desc in enumerate(frames):
        ass.render_frame(desc)
        pbar.update(i)

        if 'TEST_MOVIES' in os.environ:
            if i > 50: break

    pbar.finish()

    moviefname = moviemaker.render(args.outdir if args.outdir else os.path.dirname(BAG_FILE))
    print "wrote", moviefname

    moviemaker.cleanup()

    

