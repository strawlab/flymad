#!/usr/bin/env python

import roslib
roslib.load_manifest('flymad')

import os.path
import subprocess

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

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()

        w = self._ui.get_object("FlyMAD")
        w.connect("delete-event", rosgobject.main_quit)
        w.show_all()

        #Start ros joy_node, so that in manual control mode the joystick can be used
        self._joynode_proc = subprocess.Popen(['rosrun', 'joy', 'joy_node'])
        self._rosbag_proc = None

    def _build_ui(self):

        nodepath = "gflymad"
        package = "flymad"
        calibrationFile = "calibrationOUT"

        w = self._ui.get_object("bFlyTrax")
        w.connect("clicked", self._on_start_flytrax)

        w = self._ui.get_object("bTrackem")
        w.connect("clicked", self._on_start_trackem)

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

        port = rospy.get_param( USB_PARAM, default = USB_DEFAULT )
        w = self._ui.get_object("eUSBPort")
        w.set_text(port)

        w = self._ui.get_object("bUSBPort")
        w.connect("clicked", self._on_send_usb_port)
        self._on_send_usb_port(w)

        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bStartCalibration"),
                entry_widget=self._ui.get_object("eStartCalibration"),
                nodepath="generate_calibration",
                nodemanager=self._manager,
                package=package,
                # node_type="laser_camera_calibration.py",
                node_type="generate_calibration.py",
                args=calibrationFile + ".yaml")
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
                args=calibrationFile + ".yaml" )
                )


        self._refs.append( MyGtkButtonStartNode(
                widget=self._ui.get_object("bCalibrationResults"),
                entry_widget=self._ui.get_object("eCalibrationResults"),
                nodepath="CalibrationResults",
                nodemanager=self._manager,
                package=package,
                node_type="laser_camera_calibration.py",
                args=calibrationFile + ".filtered.yaml" )
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
                args=calibrationFile + ".filtered.yaml")
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
        subprocess.Popen(['fview', '--plugins=3'])

    def _on_start_trackem(self, widget):
        subprocess.Popen(['fview', '--plugins=1'])

    def _on_send_usb_port(self, widget):
        w = self._ui.get_object("eUSBPort")
        port = w.get_text()
        print 'setting port to',port
        rospy.set_param( USB_PARAM, port )

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
