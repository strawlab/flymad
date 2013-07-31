#!/usr/bin/env python
import math

import numpy as np
import sys
import Queue
import time

import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroPosition, Raw2dPositions
from flymad.laser_camera_calibration import save_raw_calibration_data

PER_PIXEL_SECONDS = 0.1
WAIT = 0.03

def to_plain(arr):
    if arr.ndim==1:
        return arr.tolist()
    elif arr.ndim==2:
        return [i.tolist() for i in arr]
    else:
        raise NotImplementedError('')

class Calibration:
    def __init__(self):
        rospy.init_node('calibration')
        dest = '/flymad_micro'
        self.pub = rospy.Publisher( dest+'/position', MicroPosition, tcp_nodelay=True )
        _ = rospy.Subscriber('/flymad/raw_2d_positions',
                             Raw2dPositions,
                             self.on_data)
        self.q = Queue.Queue()

    def pixels_for_dac( self, dac ):
        two, n_dacs = dac.shape
        pixels = []
        if 1:
            dur = n_dacs*PER_PIXEL_SECONDS
            print 'duration will be %.1f seconds'%dur
        for i in range(n_dacs):
            daca, dacb = dac[:,i]
            msg = MicroPosition()
            msg.posA = daca
            msg.posB = dacb
            self.pub.publish(msg)
            rospy.sleep(PER_PIXEL_SECONDS) # allow msec
            pixels.append(self.get_last_pixel(timeout=WAIT))
        pixels = np.array(pixels).T
        return pixels

    def on_data(self, msg):
        if len(msg.points):
            p = msg.points[0]
            self.q.put( (p.x, p.y) )

    def get_last_pixel(self,timeout=None):
        self._clear()
        try:
            pixel = self.q.get(True,timeout)
        except Queue.Empty:
            pixel = np.nan, np.nan
        return pixel

    def _clear(self):
        # clear queue
        while 1:
            try:
                self.q.get_nowait()
            except Queue.Empty:
                break

def main():
    fc = Calibration()
    fname = sys.argv[1]
    daca, dacb = np.mgrid[-0x3FFF:0x3FFF:50j, -0x3FFF:0x3FFF:50j]
    daca = daca.ravel().astype(np.int16)
    dacb = dacb.ravel().astype(np.int16)
    dac = np.array([daca,dacb])
    pixels = fc.pixels_for_dac( dac )
    print 'dac.shape',dac.shape
    print 'pixels.shape',pixels.shape
    save_raw_calibration_data(fname, dac, pixels)
    if 1:
        fname2 = time.strftime("%Y%m%d_%H%M%S",time.time())
        save_raw_calibration_data(fname2, dac, pixels)

if __name__=='__main__':
    main()
