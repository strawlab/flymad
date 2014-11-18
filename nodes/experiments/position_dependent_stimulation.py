#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv

from flymad.constants import LASERS_ALL_OFF, LASER2_ON

class Experiment:
    def __init__(self):
        #the IR laser is connected to 'laser2'
        self._laser_conf = rospy.Publisher('/flymad_micro/laser2/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive
        #configure the laser
        self._laser_conf.publish(enable=True,      #always enabled, we turn it on/off using /experiment/laser
                                 frequency=0,      #constant (not pulsed)
                                 intensity=1.0)    #full power

        #ensure the targeter is running so we can control the laser
        rospy.loginfo('waiting for targeter')
        rospy.wait_for_service('/experiment/laser')
        self._laser = rospy.ServiceProxy('/experiment/laser', flymad.srv.LaserState)
        self._laser(LASERS_ALL_OFF)

        #the position of the currently targeted object is sent by the targeter. In the
        #case of a single fly, the default behaviour of the targeter is to target the
        #one and only fly. If there are multiple flies you neet to instruct the targeter
        #which fly to target. The requires you recieve the raw position of all tracked
        #objects, /flymad/tracked, and decide which one is interesting by sending
        #/flymad/target_object with.
        _ = rospy.Subscriber('/targeter/targeted',
                             flymad.msg.TargetedObj,
                             self.on_targeted)

    def on_targeted(self, msg):
        #target flies in the right half of the arena
        if msg.fly_x > 400:
            rospy.loginfo('targeting fly %s' % msg.obj_id)
            self._laser(LASER2_ON)
        else:
            self._laser(LASERS_ALL_OFF)

    def run(self):
        rospy.loginfo('running')
        rospy.spin()
        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


