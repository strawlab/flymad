#!/usr/bin/env python
import numpy as np
import os, sys

import roslib; roslib.load_manifest('flymad')
from flymad.laser_camera_calibration import read_raw_calibration_data, \
     save_raw_calibration_data

def main():
    fname_in = sys.argv[1]
    fname_out = os.path.splitext(fname_in)[0] + '.filtered.yaml'
    dac_in, pixels_in = read_raw_calibration_data(fname_in)

    print 'removing all points outside radius from pixel center'
    center = np.c_[320,240].T
    r = np.sqrt(np.sum((pixels_in-center)**2,axis=0))
    valid1 = r < 300

    print 'removing all points outside near DAC zero'
    dac2 = dac_in.astype(np.int16)
    dac3 = dac2.astype(np.float)
    rd = np.sqrt(np.sum(dac3**2,axis=0))
    valid2 = rd < 16000

    valid = valid1 & valid2

    pixels = pixels_in[:,valid]
    dac = dac_in[:,valid]

    if 0:
        # useful to change some values to debug coordinates
        c = (pixels[0,:] < 400) & (pixels[1,:] < 200)
        dac[:,c]=0

    save_raw_calibration_data(fname_out, dac, pixels)

if __name__=='__main__':
    main()

