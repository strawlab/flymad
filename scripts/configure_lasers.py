#!/usr/bin/env python

import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv
import flymad.constants
import rosgobject.gtk
import rosgobject.wrappers

from gi.repository import Gtk

class UI:
    def __init__(self):
        self._w = Gtk.Window()
        self._w.connect("delete-event", rosgobject.main_quit)
        self._grid = Gtk.Grid()
        self._grid.props.row_spacing = 2
        self._grid.props.column_spacing = 5

        for i in range(3):
            self._add_laser_to_ui('Laser %d:' % i, 'flymad_micro/laser%d/configuration' % i, i)

        self._grid.attach(Gtk.Label('Targeter:'),0,i+1,1,1)

        self._srv_on = rosgobject.wrappers.GtkButtonSetServiceWidget(
                            '/experiment/laser',
                            flymad.srv.LaserState,
                            widget=Gtk.Button('All Lasers On'),
                            conv_func=lambda *a: flymad.constants.LASERS_ALL_ON)
        self._grid.attach(self._srv_on.widget,2,i+1,2,1)
        self._srv_off = rosgobject.wrappers.GtkButtonSetServiceWidget(
                            '/experiment/laser',
                            flymad.srv.LaserState,
                            widget=Gtk.Button('All Lasers Off'),
                            conv_func=lambda *a: flymad.constants.LASERS_ALL_OFF)
        self._grid.attach(self._srv_off.widget,4,i+1,2,1)

        self._w.add(self._grid)
        self._w.show_all()

    def _send_laser(self, btn, pub, cb_en, sb_freq, sb_int):
        pub.publish(enable=cb_en.get_active(),
                    frequency=sb_freq.get_value(),
                    intensity=sb_int.get_value())

    def _add_laser_to_ui(self, label, path, row):
        pub = rospy.Publisher(path, flymad.msg.LaserConfiguration)

        hb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        w = Gtk.Label(label)
        self._grid.attach(w,0,row,1,1)

        cb_en = Gtk.CheckButton("enable")
        self._grid.attach(cb_en,1,row,1,1)

        sb_freq = Gtk.SpinButton.new_with_range(0,25,0.5)
        self._grid.attach(Gtk.Label("frequency:"),2,row,1,1)
        self._grid.attach(sb_freq,3,row,1,1)

        sb_int = Gtk.SpinButton.new_with_range(0,1.0,0.05)
        self._grid.attach(Gtk.Label("intensity:"),4,row,1,1)
        self._grid.attach(sb_int,5,row,1,1)

        b = Gtk.Button("Send")
        b.connect('clicked', self._send_laser, pub, cb_en, sb_freq, sb_int)
        self._grid.attach(b,6,row,1,1)

if __name__ == "__main__":
    rosgobject.init_node("configurelasers")
    u = UI()
    rosgobject.spin()
