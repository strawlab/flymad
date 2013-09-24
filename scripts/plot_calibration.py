#!/usr/bin/env python
import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator, LinearNDInterpolator
import json
import yaml

import roslib; roslib.load_manifest('flymad')
import rospy

from flymad.laser_camera_calibration import load_calibration

def main():
    import sys
    import matplotlib.pyplot as plt
    fname = rospy.myargv()[1]
    cal = load_calibration(fname)

    if 0:
        plt.figure()
        plt.plot( cal.pixels[0,:], cal.pixels[1,:], 'b.-' )
        ax = plt.gca()
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

    if 1:
        plt.figure()
        plt.plot( cal.dac[0,:], cal.dac[1,:], 'b.-' )
        ax = plt.gca()
        ax.set_xlabel('a (DAC)')
        ax.set_ylabel('b (DAC)')

    if 1:
        fig = plt.figure()

        ax = fig.add_subplot(2,2,1)

        ax.plot( cal.pixels[0,:], cal.pixels[1,:], 'b.-' )
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        ax = fig.add_subplot(2,2,2)
        if cal.p2da is not None:
            ax.set_title('pixels->DACa')
            cax = ax.imshow(cal.p2da, origin='lower')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            fig.colorbar(cax)
        ax = fig.add_subplot(2,2,4)
        if cal.p2db is not None:
            ax.set_title('pixels->DACb')
            cax = ax.imshow(cal.p2db, origin='lower')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            fig.colorbar(cax)

    if 1:
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        if cal.d2px is not None:
            ax.set_title('DAC->pixels_x')
            da,db = np.mgrid[ -16000:16000:100j,
                               -16000:16000:100j ]
            px = cal.d2px((da,db))
            if np.sum(np.isnan(px)):
                print 'warning: setting nans to 0 for pcolor plot'
                px[ np.isnan(px) ] = 0
            plt.pcolor( da,db, px )
            ax.set_xlabel('DAC a')
            ax.set_ylabel('DAC b')
            plt.colorbar()

        ax = fig.add_subplot(2,1,2)
        if cal.d2py is not None:
            ax.set_title('DAC->pixels_y')
            da,db = np.mgrid[ -16000:16000:100j,
                               -16000:16000:100j ]
            py = cal.d2py((da,db))
            if np.sum(np.isnan(py)):
                print 'warning: setting nans to 0 for pcolor plot'
                py[ np.isnan(py) ] = 0
            plt.pcolor( da,db, py )
            ax.set_xlabel('DAC a')
            ax.set_ylabel('DAC b')
            plt.colorbar()

    plt.show()

if __name__=='__main__':
    main()
