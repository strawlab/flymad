#!/usr/bin/env python

import roslib; roslib.load_manifest('gflymad')

import os.path

import rospy
import std_msgs.msg

import rosgobject.wrappers

import rosgobject.managers
import rosgobject.gtk
from rosgobject.wrappers import *

import subprocess
from gi.repository import Gtk
import subprocess

#DIRTY: Global references to the processes launched in callbacks
#They are kept in order to send a sigterm after the process is not needed any more
rosbag_proc = None
joynode_proc = None

class UI:
    def __init__(self):
        me = os.path.dirname(os.path.abspath(__file__))
        self._ui = Gtk.Builder()
        self._ui.add_from_file(os.path.join(me,"gflymad2.glade"))

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()
        
        w = self._ui.get_object("gFlyMAD")
        w.connect("delete-event", rosgobject.main_quit)
        w.show_all()

	#Start ros joy_node, so that in manual control mode the joystick can be used
	global joynode_proc
	joynode_proc = subprocess.Popen(['rosrun', 'joy', 'joy_node'])

    def _build_ui(self):
        
        nodepath = "gflymad"
        package = "flymad"
        calibrationFile = "calibrationOUT"
        
        w = self._ui.get_object("bFlyTrax")
        w.connect("clicked", CBstartFlyTrax, None)
        
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
#                nodepath=nodepath,
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

	w = self._ui.get_object("bRosBagStart")
        w.connect("clicked", CBRosBagStart, None)

	w = self._ui.get_object("bRosBagStop")
        w.connect("clicked", CBRosBagStop, None)

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

        
def CBstartFlyTrax(widget, event, data=None):
    #print "In the CBstartFlyTrax callback!"
    subprocess.Popen(['fview', '--plugins=2'])

def CBRosBagStart(widget, event, data=None):
    #print "In the CBstartFlyTrax callback!"
    global rosbag_proc
    rosbag_proc = subprocess.Popen(['rosbag', 'record', '/flymad_target', '-o','/home/flymad/flymad_rosbag'])


def CBRosBagStop(widget, event, data=None):
    #print "In the CBstartFlyTrax callback!"
    #subprocess.Popen(['pkill', 'rosbag /flymad_target'])
    global rosbag_proc
    if(rosbag_proc != None):
        rosbag_proc.send_signal(subprocess.signal.SIGINT)
    else:
        print('You cant rosbag before starting it!')
    return False #Consume this event

if __name__ == "__main__":
    rospy.init_node("gflyMAD", anonymous=True)
    rosgobject.get_ros_thread() #ensure ros is spinning
    rosgobject.add_console_logger()
    u = UI()
    Gtk.main()
