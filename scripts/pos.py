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

try:
    a = int(sys.argv[1])
except ValueError:
    print 'input A is hex'
    a = int(sys.argv[1],16)

try:
    b = int(sys.argv[2])
except ValueError:
    print 'input B is hex'
    b = int(sys.argv[2],16)

print 'got values',a,b
print 'hex',hex(a),hex(b)

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
