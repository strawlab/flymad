#!/usr/bin/env python
import roslib; roslib.load_manifest('flymad')
import rospy

from std_msgs.msg import UInt8
from flymad.msg import MicroPosition
from flymad.srv import LaserState, LaserStateResponse
from flymad.constants import LASERS_ALL_ON, LASERS_ALL_OFF

import math

class Circler:
    def __init__(self):
        rospy.init_node('circler')
        self.pub = rospy.Publisher( 'flymad_micro/position', MicroPosition )
        self.range = min(1.0, rospy.get_param('~range', default=0.5))

        self._laser = LASERS_ALL_OFF

        _ = rospy.Subscriber('/experiment/laser',
                            UInt8,
                            self.on_laser)
        _ = rospy.Service('/experiment/laser', LaserState, self.on_laser_srv)

    def on_laser(self, msg):
        self._laser = msg.data
    def on_laser_srv(self, req):
        self._laser = req.data
        return LaserStateResponse()

    def run(self):
        r = rospy.Rate(100) # 100hz
        tf = 0.1
        while not rospy.is_shutdown():
            t = rospy.get_time()
            x = math.sin( tf*2*math.pi*t ) * ((2**14)-1) * self.range
            y = math.cos( tf*2*math.pi*t ) * ((2**14)-1) * self.range
            msg = MicroPosition()
            msg.posA = int(round(x))
            msg.posB = int(round(y))
            msg.laser = self._laser
            self.pub.publish(msg)
            r.sleep()

if __name__=='__main__':
    n=Circler()
    n.run()
