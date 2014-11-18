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


    def run(self):
        #the protocol is N repeats of on/off laser stimulation, after an initial delay
        #configuration is via ros parameters so they can be coded in a launch file
        time_delay = float(rospy.get_param('~time_delay', 5))
        time_on = float(rospy.get_param('~time_on', 10))
        time_off = float(rospy.get_param('~time_off', 10))
        repeats = int(rospy.get_param('~repeats', 2))

        rospy.loginfo('running %s %ss on / %ss off stimulations' % (repeats, time_on, time_off))

        try:
            i = 0
            rospy.sleep(time_delay)
            while (i < repeats) and (not rospy.is_shutdown()):
                rospy.loginfo('repeat %d laser on' % i)
                self._laser(LASER2_ON)
                rospy.sleep(time_on)
                rospy.loginfo('repeat %d laser off' % i)
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(time_off)
                i += 1
        except rospy.ROSInterruptException:
            #the node was cleanly killed
            pass

        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


