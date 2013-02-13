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
        self._ui.add_from_file(os.path.join(me,"gflyMAD.glade"))

        #Workaround, keep references of all rosgobject elements
        self._refs = []
        self._manager = rosgobject.managers.ROSNodeManager()
        self._build_ui()
        
        w = self._ui.get_object("gFlyMAD")
        w.connect("delete-event", rosgobject.main_quit)
        w.show_all()

    def _build_ui(self):
        
        nodeName = rospy.get_param('~nodeName') + "/"
        deviceName = "/" + rospy.get_param('~deviceName') + "/"
        
        rospy.loginfo("Starting gstroke with node name %s and device name %s" % (nodeName, deviceName) )
        
        gdeviceName = self._ui.get_object("deviceName")
        gdeviceName.set_text(rospy.get_param('~deviceName'))
        
        self._refs.append( GtkButtonStartNode(
                widget=self._ui.get_object("start"),
                nodepath="strokelitude2",
                nodemanager=self._manager,
                package="strokelitude2",
                node_type="imagenode" )
                #,args='_movie:="/media/DATA/Strokelitude2/movies/movie20121219_164901.fmf"' )
                )
        
        self._refs.append(
                GtkEntryTopicWidget(
                widget=self._ui.get_object("FPS"),
                nodepath = nodeName + "framerate",
                msgclass=std_msgs.msg.Float32,
                format_func=lambda x:"{0:5.1f} fps".format(x.data) )
                )
        
        self._refs.append(
        GtkButtonKillNode(
                widget=self._ui.get_object("stop"),
                nodepath= ''.join(nodeName.split())[:-1],
                nodemanager=self._manager )
                          )
        
        self._refs.append(
        GtkEntryParam(
                widget=self._ui.get_object("trigger"),
                nodepath=deviceName + "trigger" )
                          )
        
        self._refs.append(
        GtkSpinButtonParam(
                widget=self._ui.get_object("gain"),
                nodepath=deviceName + "gain" )
                          )
        
        self._refs.append(
        GtkSpinButtonParam(
                widget=self._ui.get_object("shutter"),
                nodepath=deviceName + "shutter" )
                          )
        
        self._refs.append(
        GtkButtonStartNode(
                widget=self._ui.get_object("glView"),
                nodepath="gl_view",
                nodemanager=self._manager,
                package="strokelitude2",
                node_type="gl_view",
                args="image_raw:={}/image_raw".format(nodeName) )
                          )
               

if __name__ == "__main__":
    rospy.init_node("gflyMAD", anonymous=True)
    rosgobject.get_ros_thread() #ensure ros is spinning
    rosgobject.add_console_logger()
    u = UI()
    Gtk.main()