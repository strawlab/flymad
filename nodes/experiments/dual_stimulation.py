#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv

from flymad.constants import LASERS_ALL_OFF, LASER2_ON, LASER1_ON

class Experiment:
    def __init__(self):
        #the IR laser is connected to 'laser1'
        self._ir_laser_conf = rospy.Publisher('/flymad_micro/laser1/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive
        #configure the laser
        self._ir_laser_conf.publish(enable=True,   #always enabled, we turn it on/off using /experiment/laser
                                 frequency=0,      #constant (not pulsed)
                                 intensity=1.0)    #full power
        #Red laser is connected to 'laser2'
        self._red_laser_conf = rospy.Publisher('/flymad_micro/laser2/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive

        #ensure the targeter is running so we can control the laser
        rospy.loginfo('waiting for targeter')
        rospy.wait_for_service('/experiment/laser')
        self._laser = rospy.ServiceProxy('/experiment/laser', flymad.srv.LaserState)
        self._laser(LASERS_ALL_OFF)


    def run(self):
        T_WAIT          = 10
        T_IR            = 5
        T_IR_AND_RED    = 5
        T_RED           = 5
        T_WAIT2         = 5

        RED_LASER = LASER2_ON
        IR_LASER  = LASER1_ON

        #the protocol is 10s wait, 5s IR only, 5s IR+red pulse, 5s red constant, 5s no stimulation
        #experiment continues until the node is killed

        rospy.loginfo('running %ss IR only, %ss IR+red, %s red only' % (T_IR, T_IR_AND_RED, T_RED))

        try:
            rospy.sleep(T_WAIT)
            while not rospy.is_shutdown():
                rospy.loginfo('IR only')
                self._laser(IR_LASER)
                rospy.sleep(T_IR)
                #configure the red laser for pulse
                self._red_laser_conf.publish(enable=True, frequency=5,intensity=1.0)
                #turn on IR and red
                rospy.loginfo('IR + Red pulse')
                self._laser(RED_LASER | IR_LASER)
                rospy.sleep(T_IR_AND_RED)
                #configure the red laser for constant
                self._red_laser_conf.publish(enable=True, frequency=0,intensity=1.0)
                #turn on red
                rospy.loginfo('Red')
                self._laser(RED_LASER)
                rospy.sleep(T_RED)
                #turn off all
                rospy.loginfo('Off')
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(T_WAIT2)

        except rospy.ROSInterruptException:
            #the node was cleanly killed
            pass

        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


