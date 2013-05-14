#! /usr/bin/python

from gi.repository import GdkX11, Gtk, Gdk, GObject

GObject.threads_init()
Gdk.threads_init()

import sys
import tempfile
import subprocess
import threading
import os.path
import shutil
import re
import time
import datetime

import numpy as np
import pandas as pd

import Image
import ImageChops

import vlc

# Create a single vlc.Instance() to be shared by (possible) multiple players.
instance = vlc.Instance("--no-snapshot-preview --snapshot-format png")

class OCRThread(threading.Thread):

    CROP = (0, 0, 245 , 35)
    THRESH = 140
    DT_RE = r"([0-9]{4})([-_ ]{1})([0-9]{2})([-_ ]{1})([0-9]{2})\ ([0-9]{2})([:_ ]{1})([0-9]{2})([:_ ]{1})([0-9]{2})([._ ]{1})([0-9]+)"

    def __init__(self, callback, key, tid):
        threading.Thread.__init__(self)
        self._cb = callback
        self._key = key
        self._tid = tid
        self._tdir = tempfile.mkdtemp(prefix='img')

    def input_image(self, fn="in.png"):
        return os.path.join(self._tdir, fn)

    def run(self):
        img = self.input_image()
        out = self.input_image("in.ppm")

        im = Image.open(img)
        im = im.crop(self.CROP)

        # Converts to black and white
        im = im.point(lambda p: p > self.THRESH and 255)  
        im.save(out)

        proc = subprocess.Popen(
                    ['gocr','-i',out,'-C','0123456789:-+.','-d','0','-a','90', '-s', '10'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
        )

        stdout,stderr = proc.communicate()

        if proc.returncode == 0:
            err = ''
            try:
                y,_,m,_,d,H,_,M,_,S,_,ms = re.match(self.DT_RE,stdout).groups()
                t = datetime.datetime(
                        int(y),int(m),int(d),
                        int(H),int(M),int(S),
                        int(float("0."+ms)*1e6), #us
                )
                #convert to seconds since epoch in UTC
                now = time.mktime(t.timetuple())+1e-6*t.microsecond
                print t
            except Exception, e:
                err = 'error parsing string %s' % stdout
                now = np.nan
        else:
            err = stderr

        shutil.rmtree(self._tdir)

        self._cb(err, now, self._key, self._tid)


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
        Gtk.Grid.__init__(self)

        self._vlc_widget = VLCWidget(*p)
        self._vlc_widget.props.hexpand = True
        self._vlc_widget.props.vexpand = True
        self.player = self._vlc_widget.player
        self.attach(self._vlc_widget,0,0,4,1)

        for i,(stock, callback) in enumerate((
            (Gtk.STOCK_MEDIA_PLAY, lambda b: self.player.play()),
            (Gtk.STOCK_MEDIA_PAUSE, lambda b: self.player.pause()),
            (Gtk.STOCK_MEDIA_STOP, lambda b: self.player.stop()),
            )):
            b = Gtk.Button(stock=stock)
            b.connect("clicked", callback)
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
        self._lbl.set_text(r.strip())
        return False

class VideoScorer(Gtk.Window):

    THREAD_REPEAT = 3

    def __init__(self):
        Gtk.Window.__init__(self)
        self.vlc = DecoratedVLCWidget()
        self.connect("key_press_event",self._on_key_press)
        self.connect("destroy", self._on_quit)
        self.add(self.vlc)

        self._annots_t = []
        self._annots_v = []
        self._annots_vn = []
        self._pending_lock = threading.Lock()
        self._pending_ocr = {}

    def main(self, fname, bagname):
        self.fname = fname
        self.bname = bagname
        self.vlc.player.set_media(instance.media_new(fname))
        self.show_all()
        Gtk.main()

    def _on_quit(self, *args):
        if self._annots_t:
            if bname and os.path.exists(bname):
                from flymadbagconv import create_df
                df = create_df(bname)
                print df

            df = pd.DataFrame(
                    {"score":self._annots_v,
                     "score_n":self._annots_vn},
                    index=self._annots_t
            )
            print df
            df.to_csv(self.fname+".csv")
        Gtk.main_quit()

    def _on_processing_finished(self, err, now, key, tid):
        with self._pending_lock:
            if err:
                GObject.idle_add(self.vlc.show_result, err)

            self._pending_ocr[tid].append(now)

            if len(self._pending_ocr[tid]) == self.THREAD_REPEAT:
                t = np.nanmin(self._pending_ocr[tid])
                if not np.isnan(t):
                    self._annots_t.append(t)
                    self._annots_v.append(key)
                    self._annots_vn.append(ord(key)) #convert ch to number
                    GObject.idle_add(self.vlc.show_result, "scored t:%s = %s" % (t,key))
                else:
                    GObject.idle_add(self.vlc.show_result, "failed to scored %s (no time calculated)" % key)

    def _on_key_press(self, widget, event):
        key = Gdk.keyval_name(event.keyval)

        #start three ocr attempts, because life is difficult and programming
        #is hard whereas cpu is cheap
        tid = int(self.vlc.player.get_time())
        self._pending_ocr[tid] = []

        if key in ["j","k"]:
            for i in range(self.THREAD_REPEAT):
                t = OCRThread(self._on_processing_finished, key, tid)
                self.vlc.player.video_take_snapshot(0,t.input_image(),0,0)
                t.start()
            return True
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
       print "You must provide a movie file name (and optionally a bag file name)"
       sys.exit(1)

    fname = sys.argv[1]

    try:
        bname = sys.argv[2]
    except IndexError:
        bname = ''

    if not os.path.isfile(fname):
        print "movie file must exist"
        sys.exit(2)

    p = VideoScorer()
    p.main(fname, bname)

