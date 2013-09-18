#!/usr/bin/env python
import numpy as np
import os, sys, time

import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.laser_camera_calibration import read_raw_calibration_data, \
     save_raw_calibration_data

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotspot', nargs=2, metavar=('hotspot_x','hotspot_y'),
                        default=(None, None), help='the (X,Y) location of the laser reflection')
    parser.add_argument('--hotspot-radius', type=float,
                        default=10.0, help='the radius of the laser reflection')

    parser.add_argument('--center', nargs=2, metavar=('center_x','center_y'),
                        default=(320, 240), help='the (X,Y) location of the arena center')
    parser.add_argument('--center-radius', type=float,
                        default=300.0, help='the radius of the laser reflection')

    fname_in = rospy.myargv()[1]
    args=rospy.myargv()[2:]

    args = parser.parse_args(args)

    fname_out = os.path.splitext(fname_in)[0] + '.filtered.yaml'
    dac_in, pixels_in = read_raw_calibration_data(fname_in)


    center_x, center_y = args.center
    center_x, center_y = map(int, (center_x, center_y))
    print 'removing all points outside radius %s of center %d,%d' % (args.center_radius,
                                                                     center_x,
                                                                     center_y )
    center = np.c_[center_x,center_y].T
    r = np.sqrt(np.sum((pixels_in-center)**2,axis=0))
    valid1 = r < args.center_radius

    print 'removing all points outside near DAC zero'
    dac2 = dac_in.astype(np.int16)
    dac3 = dac2.astype(np.float)
    rd = np.sqrt(np.sum(dac3**2,axis=0))
    valid2 = rd < 16000

    hotspot_x, hotspot_y = args.hotspot
    if hotspot_x is not None:
        assert hotspot_y is not None
        hotspot_x, hotspot_y = map(int, (hotspot_x, hotspot_y))
        print 'removing all points within radius %s of hotspot %d,%d' % (args.hotspot_radius,
                                                                         hotspot_x,
                                                                         hotspot_y )
        hotspot = np.c_[hotspot_x,hotspot_y].T
        r = np.sqrt(np.sum((pixels_in-hotspot)**2,axis=0))
        valid3 = r > args.hotspot_radius
    else:
        print 'no hotspot defined'
        valid3 = np.ones_like(pixels_in[0,:], dtype=np.bool)

    valid = (valid1 & valid2) & valid3

    pixels = pixels_in[:,valid]
    dac = dac_in[:,valid]

    if 0:
        # useful to change some values to debug coordinates
        c = (pixels[0,:] < 400) & (pixels[1,:] < 200)
        dac[:,c]=0

    save_raw_calibration_data(fname_out, dac, pixels)
    if 1:
        fname2 = time.strftime("cal_%Y%m%d_%H%M%S.filtered.out")
        save_raw_calibration_data(fname2, dac, pixels)

if __name__=='__main__':
    main()

