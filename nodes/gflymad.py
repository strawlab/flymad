#!/usr/bin/env python

import roslib
roslib.load_manifest('flymad')
import flymad.laser_camera_calibration

import os.path
import subprocess
import datetime

import rospy
import roslib.packages
import std_msgs.msg

import rosgobject.gtk
import rosgobject.managers
from rosgobject.wrappers import *

from gi.repository import Gtk

DEFAULT_PATH = os.path.expanduser("~/FLYMAD")
USB_PARAM = '/flymad_micro/port'
USB_DEFAULT = '/dev/ttyUSB0'

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

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()

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

    def _get_unfiltered_calibration_file(self):
        if self._newcalib and os.path.isfile(self._newcalib):
            {'args':self._newcalib}
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

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bTracker"),
                entry_widget=self._ui.get_object("eTracker"),
                nodepath="flymad_tracker",
                nodemanager=self._manager,
                package=package,
                node_type="tracker" )
                )


        nName = "flymad_micro"
        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bMicro"),
                entry_widget=self._ui.get_object("eMicro"),
                # nodepath=nodepath,
                nodepath=nName,
                nodemanager=self._manager,
                package=package,
                node_type=nName )
                )

        w = GtkEntryChangeParam(
                nodepath=USB_PARAM,
                create=USB_DEFAULT)
        self._refs.append(w)
        usb_grid = self._ui.get_object("grid3")
        w.widget.props.hexpand = True
        usb_grid.add(w.widget)

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


        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bCalibrationResults"),
                entry_widget=self._ui.get_object("eCalibrationResults"),
                nodepath="CalibrationResults",
                nodemanager=self._manager,
                package=package,
                node_type="plot_calibration.py",
                launch_callback=self._get_filtered_calibration_file)
                )


        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bTargeter"),
                entry_widget=self._ui.get_object("eTargeter"),
                nodepath="flymad_targeter",
                nodemanager=self._manager,
                package=package,
                node_type="targeter",
                args=calibrationFile + ".filtered.yaml")
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
        subprocess.Popen(['python', '/usr/bin/fview', '--plugins=flytrax'])

    def _on_start_trackem(self, widget):
        subprocess.Popen(['python', '/usr/bin/fview', '--plugins=trackem'])

    def _on_start_ttm(self, widget):
        subprocess.Popen(['python', '/usr/bin/fview', '--plugins=fview_head_track'])

    def _on_rosbag_start(self, widget):
        path = os.path.join( self._fcb.get_filename(), 'rosbagOut' )
        self._rosbag_proc = subprocess.Popen(['rosbag', 'record', '-a', '-o', path])

    def _on_rosbag_stop(self, widget):
        if self._rosbag_proc is not None:
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
