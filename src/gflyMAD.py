#!/usr/bin/env python

import roslib; roslib.load_manifest('gflyMAD')

import os.path

import rospy
import std_msgs.msg

import rosgobject.wrappers

import rosgobject.managers
import rosgobject.gtk
from rosgobject.wrappers import *

from gi.repository import Gtk

class UI:
    def __init__(self):
        me = os.path.dirname(os.path.abspath(__file__))
        self._ui = Gtk.Builder()
        self._ui.add_from_file(os.path.join(me,"gflyMAD2.glade"))

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()
        
        w = self._ui.get_object("gFlyMAD")
        w.connect("delete-event", rosgobject.main_quit)
        w.show_all()

    def _build_ui(self):
        
        nodepath = "flyMAD"
        package = "flyMAD"
        calibrationFile = "calibrationOUT"
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bFlyTrax"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="fview"
                ,args='_movie:="/media/DATA/Strokelitude2/movies/movie20121219_164901.fmf"' )
                )
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bMicro"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="flymad_micro" )
                )
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bStartCalibration"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="generate_calibration",
                args=calibrationFile )
                )
                
        self._refs.append( GtkButtonKillNode(
                widget=self._ui.get_object("bStopCalibration"),
                nodepath= nodepath,
                nodemanager=self._manager )
                )
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bCleanCalibration"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="filter_calibration" )
                )
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTracker"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="tracker" )
                )
                          
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("bTargeter"),
                nodepath=nodepath,
                nodemanager=self._manager,
                package=package,
                node_type="targeter" )
                )
        
        

if __name__ == "__main__":
    rospy.init_node("gflyMAD", anonymous=True)
    rosgobject.get_ros_thread() #ensure ros is spinning
    rosgobject.add_console_logger()
    u = UI()
    Gtk.main()