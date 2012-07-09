#!/usr/bin/env python
import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.msg import MicroPosition, Raw2dPositions

import math

import numpy as np
import json
import sys
import Queue

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
        self.pub = rospy.Publisher( dest+'/position', MicroPosition )
        _ = rospy.Subscriber('/flymad/raw_2d_positions',
                             Raw2dPositions,
                             self.on_data)
        self.q = Queue.Queue()

    def pixels_for_dac( self, dac ):
        two, n_dacs = dac.shape
        pixels = []
        per_pixel_seconds = 0.05
        wait = 0.03
        if 1:
            dur = n_dacs*per_pixel_seconds
            print 'duration will be %.1f seconds'%dur
        for i in range(n_dacs):
            daca, dacb = dac[:,i]
            msg = MicroPosition()
            msg.posA = daca
            msg.posB = dacb
            self.pub.publish(msg)
            rospy.sleep(per_pixel_seconds) # allow msec
            pixels.append(self.get_last_pixel(timeout=wait))
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
    daca, dacb = np.mgrid[0:0xFFFF:100j, 0:0xFFFF:100j]
    daca = daca.ravel().astype(np.uint16)
    dacb = dacb.ravel().astype(np.uint16)
    dac = np.array([daca,dacb])
    pixels = fc.pixels_for_dac( dac )
    print 'dac.shape',dac.shape
    print 'pixels.shape',pixels.shape
    to_save = {'dac':to_plain(dac),
               'pixels':to_plain(pixels.astype(np.int))}
    fd = open(fname,mode='w')
    import yaml
    json.dump(to_save,fd) # JSON is valid YAML. And faster.
    fd.close()

if __name__=='__main__':
    main()
