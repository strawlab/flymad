#!/usr/bin/env python
import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroPosition

import math

class Circler:
    def __init__(self):
        rospy.init_node('circler')
        dest = '/flymad_micro'
        self.pub = rospy.Publisher( dest+'/position', MicroPosition )

    def run(self):
        r = rospy.Rate(100) # 100hz
        tf = 0.1
        while not rospy.is_shutdown():
            t = rospy.get_time()
            x = math.sin( tf*2*math.pi*t ) * 32000 + 32000
            y = math.cos( tf*2*math.pi*t ) * 32000 + 32000
            msg = MicroPosition()
            msg.posA = int(x)
            msg.posB = int(y)
            self.pub.publish(msg)
            r.sleep()

if __name__=='__main__':
    n=Circler()
    n.run()
