#!/usr/bin/env python
"""script to set flymad DAC values to a given position

This is a workaround for stupid behavior in the rostopic
command. Specifically, the command

  rostopic pub /flymad_micro/position flymad/MicroPosition a b

could be used to acheive much the same thing as

  python pos.py a b

but in the latter case, negative numbers can also be passed.
"""
import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroPosition

import sys, time

a = int(sys.argv[1])
b = int(sys.argv[2])
print a,b

rospy.init_node('pos')
pub = rospy.Publisher( '/flymad_micro/position', MicroPosition )

for i in range(10):
    time.sleep(0.1)
    msg = MicroPosition()
    msg.posA = a
    msg.posB = b
    assert msg.posA == a
    assert msg.posB == b
    pub.publish(msg)
