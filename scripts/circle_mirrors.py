#!/usr/bin/env python
import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroPosition

import math

class Circler:
    def __init__(self):
        rospy.init_node('circler')
        self.pub = rospy.Publisher( 'flymad_micro/position', MicroPosition )

    def run(self):
        r = rospy.Rate(100) # 100hz
        tf = 0.1
        while not rospy.is_shutdown():
            t = rospy.get_time()
            x = math.sin( tf*2*math.pi*t ) * 2**14
            y = math.cos( tf*2*math.pi*t ) * 2**14
            print x,y
            msg = MicroPosition()
            msg.posA = int(round(x))
            msg.posB = int(round(y))
            self.pub.publish(msg)
            r.sleep()

if __name__=='__main__':
    n=Circler()
    n.run()
