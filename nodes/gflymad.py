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

class UI:
    def __init__(self):
        me = os.path.dirname(os.path.abspath(__file__))
        self._ui = Gtk.Builder()
        self._ui.add_from_file(
                os.path.join(
                    roslib.packages.get_pkg_dir('flymad',required=True),
                    "data","gflymad.glade")
        )

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
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTracker"),
                nodepath="flymad_tracker",
                nodemanager=self._manager,
                package=package,
                node_type="tracker" )
                )
        

        nName = "flymad_micro"
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bMicro"),
                # nodepath=nodepath,
                nodepath=nName,
                nodemanager=self._manager,
                package=package,
                node_type=nName )
                )
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bStartCalibration"),
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

        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bCleanCalibration"),
                nodepath="filter_calibration",
                nodemanager=self._manager,
                package=package,
                node_type="filter_calibration_heuristics.py",
                args=calibrationFile + ".yaml" )
                )


        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bCalibrationResults"),
                nodepath="CalibrationResults",
                nodemanager=self._manager,
                package=package,
                node_type="laser_camera_calibration.py",
                args=calibrationFile + ".filtered.yaml" )
                )
        
                                  
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTargeter"),
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

        self.killer = Killer()
        w = self._ui.get_object("bKillObjects")
        w.connect("clicked", self.killer.cb_kill_all_tracked_objects)

        w = self._ui.get_object("bRosBagStart")
        w.connect("clicked", self._on_rosbag_start)

        w = self._ui.get_object("bRosBagStop")
        w.connect("clicked", self._on_rosbag_stop)

        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bStartManualControll"),
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
        subprocess.Popen(['fview', '--plugins=2'])

    def _on_rosbag_start(self, widget):
        self._rosbag_proc = subprocess.Popen(['rosbag', 'record', '-a', '-o','/home/flymad/flymad_rosbag/rosbagOut'])

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
