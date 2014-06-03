#! /usr/bin/python

from gi.repository import GdkX11, Gtk, Gdk, GObject

GObject.threads_init()
Gdk.threads_init()

import sys
import random
import tempfile
import subprocess
import threading
import os.path
import shutil
import re
import time
import datetime
import glob
import zipfile
import argparse
import collections

import numpy as np
import pandas as pd

import Image
import ImageChops

import roslib; roslib.load_manifest('flymad')
import flymad.vlc as vlc
import flymad.conv as bagconv
import flymad.filename_regexes as filename_regexes

# Create a single vlc.Instance() to be shared by (possible) multiple players.
instance = vlc.Instance("--no-snapshot-preview --snapshot-format png")

class OCRThread(threading.Thread):

    #set to zero to let gocr auto-threshold the image
    THRESH = 0#140

    #spaces are sometime spuriously detected, dots are often not detected. Drop
    #both and adjust the regex to not need them
    #2013-08-1411:00:30092675+02:00
    DT_CROP = (0, 0, 245 , 35)
    DT_RE = r"([0-9]{4})([_ ]{1})([0-9]{2})([_ ]{1})([0-9]{2})([0-9]{2})([:_ ]{1})([0-9]{2})([:_ ]{1})([0-9]{2})([._ ]{0,1})([0-9]+)"

    T_CROP = (85, 0, 245 , 35)
    T_RE = r"([0-9]{2})([:_ ]{1})([0-9]{2})([:_ ]{1})([0-9]{2})([._ ]{0,1})([0-9]+)"

    F_CROP = (0, 35, 150, 55)
    F_RE = r"([0-9]+)"

    MODE_NORMAL = 'normal'
    MODE_FORCE_DATE = 'force date'
    MODE_FRAMENUMBER = 'framenumber'

    CROPS = {MODE_NORMAL:DT_CROP,
             MODE_FORCE_DATE:T_CROP,
             MODE_FRAMENUMBER:F_CROP}
    RES = {MODE_NORMAL:DT_RE,
           MODE_FORCE_DATE:T_RE,
           MODE_FRAMENUMBER:F_RE}

    def __init__(self, callback, key, tid, mode, force_date):
        threading.Thread.__init__(self)
        self._cb = callback
        self._key = key
        self._tid = tid
        self._tdir = tempfile.mkdtemp(prefix='img')
        self._mode = mode

        self._force_date = force_date
        self._crop = self.CROPS[mode]
        self._re = re.compile(self.RES[mode])

    def input_image(self, fn="in.png"):
        return os.path.join(self._tdir, fn)

    def run_regex(self, stdout):

        t = None
        if self._mode == self.MODE_NORMAL:
            y,_,m,_,d,H,_,M,_,S,_,ms = self._re.match(stdout).groups()
        elif self._mode == self.MODE_FORCE_DATE:
            y,m,d = self._force_date
            H,_,M,_,S,_,ms = self._re.match(stdout).groups()
        elif self._mode == self.MODE_FRAMENUMBER:
            t, = self._re.match(stdout).groups()
        else:
            raise Exception("Unknown Mode")

        if t is None:
            t = datetime.datetime(
                    int(y),int(m),int(d),
                    int(H),int(M),int(S),
                    int(float("0."+ms)*1e6), #us
            )

        return t


    def run(self):
        img = self.input_image()
        out = self.input_image("in.ppm")

        im = Image.open(img)
        im = im.crop(self._crop)

        # Converts to black and white
        if self.THRESH > 0:
            im = im.point(lambda p: p > self.THRESH and 255)

        im.save(out)

        proc = subprocess.Popen(
                    ['gocr','-i',out,'-C','0123456789:+.','-d','0','-a','80', '-s', '10'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
        )

        stdout,stderr = proc.communicate()

        if proc.returncode == 0:
            err = ''
            try:
                #remove spaces
                stdout = stdout.replace(' ','')
                dt = self.run_regex(stdout)
                if self._mode == self.MODE_FRAMENUMBER:
                    now = int(dt)
                else:
                    #convert to seconds since epoch in UTC
                    now = time.mktime(dt.timetuple())+1e-6*dt.microsecond
                print stdout.replace('\n','')," = ",dt, now
            except Exception, e:
                err = 'error parsing string %s' % stdout
                print err
                now = np.nan
        else:
            err = stderr

        self._cb(err, now, out, self._key, self._tid)

        shutil.rmtree(self._tdir)


class VLCWidget(Gtk.DrawingArea):

    def __init__(self, *p):
        Gtk.DrawingArea.__init__(self)
        self.player = instance.media_player_new()
        def handle_embed(*args):
            self.player.set_xwindow(self.props.window.get_xid())
            return True
        self.connect("map", handle_embed)
        self.set_size_request(1024, 384)

class DecoratedVLCWidget(Gtk.Grid):

    def __init__(self, *p):
        Gtk.Grid.__init__(self,margin=5,column_spacing=5,row_spacing=5)

        self._vlc_widget = VLCWidget(*p)
        self._vlc_widget.props.hexpand = True
        self._vlc_widget.props.vexpand = True
        self.player = self._vlc_widget.player
        self.attach(self._vlc_widget,0,0,5,1)

        for i,(stock, func) in enumerate((
            (Gtk.STOCK_MEDIA_PLAY, self.player.play),
            (Gtk.STOCK_MEDIA_PAUSE, self.player.pause),
            (Gtk.STOCK_MEDIA_NEXT, self.player.next_frame),
            (Gtk.STOCK_MEDIA_STOP, self.player.stop),
            )):
            b = Gtk.Button(stock=stock)
            b.connect("clicked", self._on_click_player_command, func)
            b.props.hexpand = False
            b.props.halign = Gtk.Align.START
            self.attach(b,i,1,1,1)

        self._scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL,0,1,0.01)
        self._scale.props.hexpand = True
        self._scale.props.draw_value = False
        self._scale_changing_by_us = False
        self._scale.connect("value-changed", self._slider_changed)
        self.attach(self._scale,i+1,1,1,1)

        self._lbl = Gtk.Label("")
        self._lbl.props.hexpand = True
        self._lbl.props.halign = Gtk.Align.END
        self._lbl.props.xalign = 1.0
        self.attach(self._lbl,1,2,4,1)

        GObject.timeout_add(50,self._update_slider)

    def _on_click_player_command(self, btn, func):
        func()
        self._update_slider()

    def _slider_changed(self, scale):
        if not self._scale_changing_by_us:
            self.player.set_position(self._scale.get_value())

    def _update_slider(self, *args):
        pos = max(0,self.player.get_position())
        self._scale_changing_by_us = True
        self._scale.set_value(pos)
        self._scale_changing_by_us = False
        return True

    def show_result(self, r):
        self._lbl.set_markup(r.strip())
        return False

class VideoScorer(Gtk.Window):

    THREAD_REPEAT = 3
    #maps keys to which colum in the dataframe they are stored
    KEYS = {
        "a":"as",
        "s":"as",
        "z":"zx",
        "x":"zx",
        "c":"cv",
        "v":"cv",
        "q":"qw",
        "w":"qw",
    }
    #which of the pair as defined above means 'OFF'
    KEY_DEFAULTS = {
        "as":"s",
        "zx":"x",
        "cv":"v",
        "qw":"w",
    }


    def __init__(self, mode, force_date, out_fname):
        Gtk.Window.__init__(self)
        self.blind_fname = None
        self.out_fname = out_fname
        self.vlc = DecoratedVLCWidget()
        self.connect("key-press-event",self._on_key_press)
        self.connect("delete-event", self._on_quit)
        self.connect("destroy", Gtk.main_quit)
        self.add(self.vlc)

        self._faildir = tempfile.mkdtemp(prefix='img')
        self._failures = {}

        #annots[time] = value
        self._annots = collections.OrderedDict()

        self._pending_lock = threading.Lock()
        self._pending_ocr = {}

        self._mode = mode

        if force_date:
            self._force_date = map(int,args.force_date.split('/'))
        else:
            self._force_date = tuple()

    def main(self, fname, bagname, set_title):
        self.fname = fname
        self.bname = bagname
        self.blind_fname = tempfile.mktemp(suffix='.mp4') # generate random filename
        os.symlink(os.path.abspath(fname), self.blind_fname)
        self.vlc.player.set_media(instance.media_new(self.blind_fname))
        if set_title:
            self.set_title(os.path.basename(fname))
        self.show_all()
        Gtk.main()

    def _on_quit(self, *args):
        self.vlc.player.stop()

        #Update the UI
        self.vlc.show_result("merging bag file, please wait")
        while Gtk.events_pending():
            Gtk.main_iteration()

        cols = set(self.KEYS.values())

        dfd = {k:[] for k in cols}
        if self._mode == OCRThread.MODE_FRAMENUMBER:
            dfd["framenumber"] = []

        idx = []
        for t in sorted(self._annots):
            idx.append(t)
            key = self._annots[t]
            thiscol = self.KEYS[key]

            for col in cols:
                dfd[col].append( key if thiscol == col else np.nan )

            if self._mode == OCRThread.MODE_FRAMENUMBER:
                dfd["framenumber"].append(t)

        df = pd.DataFrame(
                dfd,
                index=(np.array(idx)*bagconv.SECOND_TO_NANOSEC).astype(np.int64)
        )

        if self.bname:
            bdf = bagconv.create_df(self.bname)
            final = pd.concat((df,bdf),axis=1)
        else:
            final = df

#        print "annot index", df.index,
#        print " unique ",df.index.is_unique
#        print "bag index", bdf.index,
#        print " unique ", bdf.index.is_unique
#        print "annots", self._annots
#        print "dfd", dfd
#        print "idx", idx

        filled = final.fillna(method='pad')

        filled.to_csv(self.out_fname)

        print "saved CSV file"

        if self._failures:
            dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.QUESTION,
                Gtk.ButtonsType.YES_NO, "Incomplete Scoring")
            dlg.format_secondary_text(
                "%d frames failed to score. Do you wish to save these frames "\
                "for analysis?" % len(self._failures))
            rsp = dlg.run()
            if rsp == Gtk.ResponseType.YES:
                with zipfile.ZipFile(self.fname+"_scorefail.zip", 'w') as myzip:
                    for v in self._failures.values():
                        myzip.write(v)
            dlg.destroy()

        if self.blind_fname is not None:
            os.unlink(self.blind_fname)
            self.blind_fname = None
        #keep quitting
        return False

    def _on_processing_finished(self, err, now, ocrimg, key, tid):
        with self._pending_lock:

            self._pending_ocr[tid].append(now)

            if len(self._pending_ocr[tid]) == self.THREAD_REPEAT:
                t = np.nanmin(self._pending_ocr[tid])
                annot = self.KEYS[key]
                if not np.isnan(t):
                    self._annots[t] = key
                    if self._mode == OCRThread.MODE_FRAMENUMBER:
                        res = t
                    else:
                        res = datetime.datetime.fromtimestamp(t)
                    GObject.idle_add(self.vlc.show_result, "%s = '%s'" % (res,key))
                else:
                    GObject.idle_add(self.vlc.show_result, "failed to scored '%s' (no time calculated)" % key)

                    outimg = os.path.join(self._faildir,"%s.ppm" % tid)
                    shutil.copyfile(ocrimg,outimg)
                    self._failures[tid] = outimg

    def _on_key_press(self, widget, event):
        key = Gdk.keyval_name(event.keyval)

        #start three ocr attempts, because life is difficult and programming
        #is hard whereas cpu is cheap
        tid = int(self.vlc.player.get_time())
        self._pending_ocr[tid] = []

        if key in self.KEYS:
            for i in range(self.THREAD_REPEAT):
                t = OCRThread(self._on_processing_finished, key, tid, self._mode, self._force_date)
                self.vlc.player.video_take_snapshot(0,t.input_image(),0,0)
                t.start()
            return True
        elif key  == 'Delete':
            try:
                key, val = self._annots.popitem()
                self.vlc.show_result("deleted annotation %s = '%s' (%d remain)" % (key,val,len(self._annots)))
            except KeyError:
                pass #empty
            return True

        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='movie_or_directory_of_movies', nargs=1)
    parser.add_argument('--force-date', metavar='YYYY/MM/DD',
                help='force the date of the events, only extract the time '\
                     'from the video')
    parser.add_argument('--bagdir', type=str, default=None)
    parser.add_argument('--outdir', type=str, default=None,
                        help='directory for output .csv files')
    parser.add_argument('--framenumber', action='store_true',
                help='extract the framenumber from the video instead')
    parser.add_argument('--no-merge-bags', action='store_true',
                help='dont attempt to merge with bag files')
    parser.add_argument('--max-trial', type=int, default=1000,
                help='maximum trial number to score')
    parser.add_argument('--set-title', action='store_true',
                help='set the window title to the current movie '\
                     'warning: this might bias your scoring')
    parser.add_argument('--skip-existing', action='store_true',
                help='do not re-score mp4s that have already been scored')

    args = parser.parse_args()

    mode = OCRThread.MODE_NORMAL

    if args.force_date:
        try:
            y,m,d = map(int,args.force_date.split('/'))
            mode = OCRThread.MODE_FORCE_DATE
        except:
            parser.error("could not parse date string: %s" % args.force_date)
    elif args.framenumber:
        mode = OCRThread.MODE_FRAMENUMBER

    if not os.path.exists('/usr/bin/gocr'):
        raise RuntimeError('you need to install gocr')

    directory = args.directory[0]
    if args.bagdir is None:
        args.bagdir = directory

    if args.outdir is None:
        args.outdir = directory

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        if not os.path.isdir(args.outdir):
            raise RuntimeError('out directory is not a directory')

    if os.path.isdir(directory):
        inputmp4s = glob.glob(os.path.join(directory,"*.mp4"))
        if args.bagdir is None:
            bagdir = directory
        else:
            bagdir = args.bagdir
    elif os.path.isfile(directory):
        inputmp4s = [directory]
        if args.bagdir is None:
            bagdir = '/dev/null'
        else:
            bagdir = os.path.dirname(directory)
    else:
        sys.exit(1)

    inputbags = glob.glob(os.path.join(bagdir,"*.bag"))
    if len(inputbags)==0:
        print 'no bag files found in %r' % bagdir

    random.shuffle(inputmp4s)

    real_input_mp4s = []
    for mp4 in inputmp4s:
        fname = mp4
        base_fname = os.path.basename(fname)
        out_fname = os.path.join(args.outdir, base_fname+'.csv')
        if os.path.exists(out_fname):
            if args.skip_existing:
                print "skipping", fname
                continue
            else:
                print 'will overwrite output file',out_fname
        real_input_mp4s.append( mp4 )

    inputmp4s = real_input_mp4s
    print 'will score %d mp4 movies'%(len(inputmp4s),)

    for mp4 in inputmp4s:
        fname = mp4
        base_fname = os.path.basename(fname)
        out_fname = os.path.join(args.outdir, base_fname+'.csv')

        if args.no_merge_bags:
            bname = None
        else:
            try:
                mp4time = filename_regexes.parse_date(mp4)
            except filename_regexes.RegexError, e:
                print "error: incorrectly named mp4 file?", mp4
                continue

            bname = None
            if inputbags:
                best_diff = np.inf
                for bag in inputbags:
                    try:
                        bagtime = filename_regexes.parse_date(bag)
                    except filename_regexes.RegexError, e:
                        print "error: incorrectly named bag file?", bag
                        continue
                    this_diff = abs(time.mktime(bagtime)-time.mktime(mp4time))
                    if this_diff < best_diff:
                        bname = bag
                        best_diff = this_diff
                    else:
                        continue
                assert best_diff < 10.0
                assert os.path.exists(bname)

        assert os.path.exists(fname)

        if not os.path.isfile(fname):
            print "movie file must exist. give input dir, not dir/"
            sys.exit(2)

        p = VideoScorer(mode, args.force_date, out_fname)
        p.main(fname, bname, args.set_title)

