#!/usr/bin/env python

import roslib
import roslib.packages
roslib.load_manifest('flymad')
import flymad.laser_camera_calibration
import flymad.refined_utils
import flymad.msg

import os.path
import subprocess
import datetime
import glob
import threading
import time
import json
import urllib2

from distutils.version import StrictVersion

import rospy
import roslib.packages
import std_msgs.msg

import rosgobject.gtk
import rosgobject.managers
from rosgobject.wrappers import *

from gi.repository import Gtk, GLib, GObject

DEFAULT_PATH = os.path.expanduser("~/FLYMAD")
USB_PARAM = '/flymad_micro/port'
USB_DEFAULT = '/dev/ttyUSB0'

class VersionChecker(threading.Thread, GObject.GObject):

    __gsignals__ =  {
            "version": (
                GObject.SignalFlags.RUN_LAST, None, [
                str]),
            }

    def __init__(self):
        threading.Thread.__init__(self, name="version_check")
        self.daemon = True
        GObject.GObject.__init__(self)

    def _emit_version(self, s):
        GObject.idle_add(GObject.GObject.emit,self,"version",s)

    def check(self, check_url):
        self._url = check_url
        self.start()

    def run(self):
        time.sleep(2)
        try:
            req = urllib2.Request(self._url)
            req.add_header('User-Agent', 'gflymad/%s'%flymad.__version__)
            resp = urllib2.urlopen(req)
            content = resp.read()
            self._emit_version(content)
        except:
            self._emit_version('')

class MyGtkButtonStartNode(GtkButtonStartNode):
    def __init__(self,*args,**kwargs):
        entry_widget = kwargs.pop('entry_widget')
        super(MyGtkButtonStartNode,self).__init__(*args,**kwargs)
        if entry_widget is not None:
            entry_widget.set_text('{node_type}'.format(**kwargs))

class UI:
    def __init__(self):
        me = os.path.dirname(os.path.abspath(__file__))
        self._ui = Gtk.Builder()
        self._ui.add_from_file(
                os.path.join(
                    roslib.packages.get_pkg_dir('flymad',required=True),
                    "data","gflymad.glade")
        )

        self._vercheck = VersionChecker()
        self._vercheck.connect("version", self._on_got_version)

        self._fcb = self._ui.get_object("filechooserbutton1")
        self._fcb.set_filename(DEFAULT_PATH)
        self._fcb.props.tooltip_text = DEFAULT_PATH

        self._newcalib = ''

        self._lbl_calib = self._ui.get_object("lbl_calib")
        self._lbl_calib_f = self._ui.get_object("lbl_calib_f")

        #get the calibration file
        c = flymad.laser_camera_calibration.get_calibration_file_path(
                    default=rospy.get_param('flymad/calibration', None))
        self._cfcb = self._ui.get_object("filechooserbutton2")
        self._cfcb.set_filename(c)
        self._cfcb.props.tooltip_text = c
        self._cfcb.connect('file-set', self._on_calib_file_set)
        self._short_set_lbl(self._lbl_calib_f, c)

        #keep stats of TTM
        self._stats = {
                1:flymad.refined_utils.StatsManager(1),
                5:flymad.refined_utils.StatsManager(5),
                30:flymad.refined_utils.StatsManager(30),
        }
        _ = rospy.Subscriber('/flymad/laser_head_delta',
                             flymad.msg.HeadDetect,
                             self._on_head_delta)
        GLib.timeout_add_seconds(1, self._update_stats)

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()

        self._window = self._ui.get_object("FlyMAD")

        self._ksb = GtkButtonKillNode(
                        self._ui.get_object("bkillExperiment"),
                        self._manager,
                        "/experiment"
        )

        w = self._ui.get_object("FlyMAD")
        w.connect("delete-event", rosgobject.main_quit)
        w.show_all()

        #Start ros joy_node, so that in manual control mode the joystick can be used
        self._joynode_proc = subprocess.Popen(['rosrun', 'joy', 'joy_node'])
        self._rosbag_proc = None

        cdir = os.path.join(GLib.get_user_config_dir(), 'flymad')
        if not os.path.isfile(os.path.join(cdir,'no_version_check')):

            if os.path.isfile(os.path.join(cdir,'prefer_development_version')):
                url = "http://updates.flymad.strawlab.org/development/"
            else:
                url = "http://updates.flymad.strawlab.org/stable/"

            self._vercheck.check(url)

    def _on_got_version(self, t, version):
        if not version:
            return

        try:
            dat = json.loads(version)
            if StrictVersion(flymad.__version__) < StrictVersion(dat['version']):

                grid = self._ui.get_object("grid1")
                grid.insert_row(0)
                bar = Gtk.InfoBar()
                grid.attach(bar, 0,0,2,1)
                bar.add_button(Gtk.STOCK_CLOSE, Gtk.ResponseType.OK)
                bar.connect('response', lambda _b, _r: _b.hide())
                bar.set_message_type(Gtk.MessageType.INFO)
                label = Gtk.Label()
                label.set_markup('A new version of FlyMAD is available. '
                                 '<a href="%s">Click here</a> to learn more' % (
                                 dat["url"],))
                bar.get_content_area().pack_start(label, False, False, 0)
                bar.show_all()

        except Exception, e:
            print "ERROR checking version: %s" % e

    def _on_head_delta(self, msg):
        for v in self._stats.itervalues():
            v.process(msg)

    def _update_stats(self):
        for time,stat in self._stats.iteritems():
            pt,ac = stat.get_stats()
            #the widgets are named according to target and time
            #lbl_stat_pt_2_30
            #            ^  ^--- time
            #            |------ target 
            for trg in flymad.refined_utils.TTM_NAMES:
                lbl = self._ui.get_object("lbl_stat_pt_%d_%d" % (trg,time))
                lbl.set_text("%.0f" % (1.0/pt[trg]))
            for trg in flymad.refined_utils.TTM_NAMES:
                lbl = self._ui.get_object("lbl_stat_ac_%d_%d" % (trg,time))
                lbl.set_text("%.2f" % (100.0*ac[trg]))

        return True

    def _short_set_lbl(self, lbl, txt):
        h = os.path.expanduser('~')
        txt = txt.replace(h,'~')
        lbl.set_markup('<i>%s</i>' % txt)

    def _on_calib_file_set(self, *args):
        self._short_set_lbl(
                    self._lbl_calib_f,
                    self._cfcb.get_filename())

    def _get_filtered_calibration_file(self):
        return {'args':self._cfcb.get_filename()}

    def _get_targeter_and_calib(self):
        return {'node_type':self._targeters.get_active_id(),
                'args':self._cfcb.get_filename()}

    def _get_unfiltered_calibration_file(self):
        if self._newcalib and os.path.isfile(self._newcalib):
            return {'args':self._newcalib}
        else:
            self._short_set_lbl(
                        self._lbl_calib,
                        "create a new calibration file")
            raise Exception("create a new calibration file")

    def _get_new_calibration_file(self):
        newfn =  flymad.laser_camera_calibration.get_calibration_file_path(
                    create=True,
                    suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        #create the file
        with open(newfn, 'w') as f:
            pass

        self._newcalib = newfn
        self._short_set_lbl(
                    self._lbl_calib,
                    newfn)

        return {'args':newfn}

    def _build_ui(self):

        nodepath = "gflymad"
        package = "flymad"
        calibrationFile = "calibrationOUT"

        w = self._ui.get_object("bFlyTrax")
        w.connect("clicked", self._on_start_flytrax)

        w = self._ui.get_object("bTrackem")
        w.connect("clicked", self._on_start_trackem)

        w = self._ui.get_object("bTTM")
        w.connect("clicked", self._on_start_ttm)

        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTracker"),
                nodepath="flymad_tracker",
                nodemanager=self._manager,
                package=package,
                node_type="tracker" )
                )
        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopTracker"),
                nodepath= "/flymad_tracker",
                nodemanager=self._manager )
                )

        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bMicro"),
                nodepath="flymad_micro",
                nodemanager=self._manager,
                package=package,
                node_type="flymad_micro")
                )
        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopMicro"),
                nodepath= "/flymad_micro",
                nodemanager=self._manager )
                )

        w = GtkEntryChangeParam(
                nodepath=USB_PARAM,
                create=USB_DEFAULT)
        self._refs.append(w)
        usb_grid = self._ui.get_object("grid3")
        w.widget.props.hexpand = True
        usb_grid.add(w.widget)

        tg = self._ui.get_object("grid6")
        w = GtkSpinButtonParam(
                nodepath="/ttm/px",
                min=-1.0,
                max=1.0,
                step=0.01)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,1,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/py",
                min=-1.0,
                max=1.0,
                step=0.01)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,2,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/pv",
                min=0.0,
                max=0.2,
                step=0.0001)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,3,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/latency",
                min=0.0,
                max=0.1,
                step=0.001)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,4,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/ttm_gyro_axes_flip",
                min=0,
                max=1,
                step=1)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,5,2,1)

        w = GtkSpinButtonParam(
                nodepath="/ttm/headtrack_downsample",
                min=1,
                max=4,
                step=1)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,6,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/headtrack_mincontourarea",
                min=500,
                max=7000,
                step=500)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,7,2,1)

        w = GtkSpinButtonParam(
                nodepath="/ttm/headtrack_checkflipped",
                min=0,
                max=1,
                step=1)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,8,2,1)
        w = GtkSpinButtonParam(
                nodepath="/ttm/headtrack_usegpu",
                min=0,
                max=1,
                step=1)
        self._refs.append(w)
        tg.attach(w.widget,
                  2,9,2,1)

        self._targeters = self._ui.get_object("comboboxtext1")
        ndir = os.path.join(roslib.packages.get_pkg_dir('flymad'), 'nodes')
        for n in glob.glob('%s/*targeter*' % ndir):
            nn = os.path.basename(n)
            if nn.endswith('~'):
                continue
            self._targeters.append(nn,nn)
        self._targeters.set_active_id('targeter')

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bStartCalibration"),
                entry_widget=self._ui.get_object("eStartCalibration"),
                nodepath="generate_calibration",
                nodemanager=self._manager,
                package=package,
                node_type="generate_calibration.py",
                launch_callback=self._get_new_calibration_file)
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopCalibration"),
                nodepath= "/generate_calibration",
                nodemanager=self._manager )
                )

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bCleanCalibration"),
                entry_widget=self._ui.get_object("eCleanCalibration"),
                nodepath="filter_calibration",
                nodemanager=self._manager,
                package=package,
                node_type="filter_calibration_heuristics.py",
                launch_callback=self._get_unfiltered_calibration_file)
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopTTM"),
                nodepath= "/fview_ttm",
                nodemanager=self._manager )
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopFlytrax"),
                nodepath= "/fview_flytrax",
                nodemanager=self._manager )
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopTrackem"),
                nodepath= "/fview_trackem",
                nodemanager=self._manager )
                )


        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bCalibrationResults"),
                entry_widget=self._ui.get_object("eCalibrationResults"),
                nodepath="CalibrationResults",
                nodemanager=self._manager,
                package=package,
                node_type="plot_calibration.py",
                launch_callback=self._get_filtered_calibration_file)
                )


        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTargeter"),
                nodepath="flymad_targeter",
                nodemanager=self._manager,
                package=package,
                launch_callback=self._get_targeter_and_calib)
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopTargeter"),
                nodepath= "/flymad_targeter",
                nodemanager=self._manager )
                )

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bViewer"),
                entry_widget=self._ui.get_object("eViewer"),
                nodepath="flymad_viewer",
                nodemanager=self._manager,
                package=package,
                node_type="viewer",
                launch_callback=self._get_filtered_calibration_file)
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopViewer"),
                nodepath= "/flymad_viewer",
                nodemanager=self._manager )
                )

        self.killer = Killer()
        w = self._ui.get_object("bKillObjects")
        w.connect("clicked", self.killer.cb_kill_all_tracked_objects)

        w = self._ui.get_object("bRosBagStart")
        w.connect("clicked", self._on_rosbag_start)

        w = self._ui.get_object("bRosBagStop")
        w.connect("clicked", self._on_rosbag_stop)

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bStartManualControll"),
                entry_widget=self._ui.get_object("eStartManualControll"),
                nodepath="flymad_joy",
                nodemanager=self._manager,
                package=package,
                node_type="flymad_joy")
                )

        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopManualControll"),
                nodepath= "/flymad_joy",
                nodemanager=self._manager )
                )

    def _on_start_flytrax(self, widget):
        subprocess.Popen(['python', '/usr/bin/fview', '__name:=fview_flytrax', '--plugins=flytrax'])

    def _on_start_trackem(self, widget):
        subprocess.Popen(['python', '/usr/bin/fview', '__name:=fview_trackem', '--plugins=trackem,fview_draw_geom'])

    def _on_start_ttm(self, widget):
        subprocess.Popen(['python', '/usr/bin/fview', '__name:=fview_ttm', '--plugins=fview_head_track'])

    def _on_rosbag_start(self, widget):
        path = os.path.join( self._fcb.get_filename(), 'rosbagOut' )
        name = self._manager.random_node_name(10, 'rosbag')
        self._ui.get_object('eRosBagStart').set_text('/'+name)
        self._rosbag_proc = subprocess.Popen(['rosbag', 'record', '__name:=%s' % name, '-a', '-o', path])

    def _on_rosbag_stop(self, widget):
        if self._rosbag_proc is not None:
            self._ui.get_object('eRosBagStart').set_text('')
            self._rosbag_proc.send_signal(subprocess.signal.SIGINT)

class Killer:
    def __init__(self):
        self.pub = rospy.Publisher( '/flymad/kill_all',
                                    std_msgs.msg.UInt8,)
    def cb_kill_all_tracked_objects(self,widget):
        self.pub.publish(True)

if __name__ == "__main__":
    rospy.init_node("gflyMAD", anonymous=True)
    rosgobject.get_ros_thread() #ensure ros is spinning
    rosgobject.add_console_logger()
    u = UI()
    Gtk.main()
