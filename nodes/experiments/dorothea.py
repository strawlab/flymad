#!/usr/bin/env python
import itertools

import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv
import geometry_msgs.msg

from flymad.constants import LASERS_ALL_OFF, LASER2_ON
from motmot.fview.utils import lineseg_circle

from transitions import State, Machine, Transition

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

class OrderedStateMachine(Machine):

    def __init__(self, *args, **kwargs):
        Machine.__init__(self, *args, **kwargs)

        if len(self.states) < 2:
            raise ValueError('OrderedStateMachine must be defined at constuction time')

        def add_transition(_from,_to):
            self.add_transition('%s_to_%s' % (_from,_to), _from, _to)

        #auto define transitions between states
        state_names = self.states.keys()
        for states in pairwise(state_names):
            add_transition(*states)
        #add a transition from the last to the initial state
        add_transition(state_names[-1], self._initial)
        #add a transition from the last to the first non-initial state
        add_transition(state_names[-1],[s for s in state_names if s != self._initial][0])

    def next_state(self, loop_includes_initial=False):
        sn = self.states.keys()
        if not loop_includes_initial and self.current_state.name != self._initial:
            sn.remove(self._initial)

        sn = itertools.cycle(sn)
        for s in sn:
            if self.current_state.name == s:
                getattr(self.model,"%s_to_%s" % (self.current_state.name,sn.next()))()
                break


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

        self._flyx, self._flyy = 0,0
        #the position of the currently targeted object is sent by the targeter. In the
        #case of a single fly, the default behaviour of the targeter is to target the
        #one and only fly. If there are multiple flies you neet to instruct the targeter
        #which fly to target. The requires you recieve the raw position of all tracked
        #objects, /flymad/tracked, and decide which one is interesting by sending
        #/flymad/target_object with.
        _ = rospy.Subscriber('/targeter/targeted',
                             flymad.msg.TargetedObj,
                             self.on_targeted)

        self._pubpts = rospy.Publisher('/draw_geom/poly',
                                    geometry_msgs.msg.Polygon,
                                    latch=True)

        #set up the states of this experiment
        s1 = State('laser_on', on_enter=['on_enter_laser_on'])
        s2 = State('laser_off', on_enter=['on_enter_laser_off'])
        s3 = State('tile', on_enter=['on_enter_tile'])
        self._machine = OrderedStateMachine(self, states=[s1,s2,s3])

    def on_targeted(self, msg):
        self._flyx = msg.fly_x
        self._flyy = msg.fly_y

    #these are the functions called on state transision
    def on_enter_laser_on(self, *args):
        self._laser(LASER2_ON)
    def on_enter_laser_off(self, *args):
        self._laser(LASERS_ALL_OFF)
    def on_enter_tile(self, *args):
        segs = lineseg_circle(self._flyx,self._flyy,50)
        pts = []
        for seg in segs:
            pts.append( geometry_msgs.msg.Point32(seg[0],seg[1],0) )
        self._pubpts.publish(pts)

    def run(self):
        LASER_ON_TIME   = 3
        LASER_OFF_TIME  = 3
        BOX_TIME        = 6

        rospy.loginfo('running')

        self._machine.next_state()
        rospy.sleep(LASER_ON_TIME)
        self._machine.next_state()
        rospy.sleep(LASER_OFF_TIME)
        self._machine.next_state()
        rospy.sleep(BOX_TIME)
        
        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


