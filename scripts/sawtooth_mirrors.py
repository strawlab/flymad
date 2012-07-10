#!/usr/bin/env python
import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroVelocity

import serial
import time

def sawtooth():
    rospy.init_node('sawtooth')
    dest = '/flymad_micro'
    pub = rospy.Publisher( dest+'/velocity', MicroVelocity )
    time.sleep(0.2) # give some time. (Seems needed so published message is not lost.)

    msg = MicroVelocity()
    msg.velA = 2**14
    msg.velB = -40000.0
    pub.publish(msg)
    print 'published velocity...', msg

if __name__=='__main__':
    sawtooth()
