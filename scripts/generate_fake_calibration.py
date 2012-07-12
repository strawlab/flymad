import numpy as np
import sys

import roslib; roslib.load_manifest('flymad')
from flymad.laser_camera_calibration import save_raw_calibration_data

class FakeCalibration:
    def __init__(self):
        self.a = 1.0
    def pixels_for_dac(self,dac):
        # dac is a 2xN array where N is number of points
        assert dac.ndim==2
        assert dac.shape[0]==2
        # first do a small linear transform
        R = np.array( [[0.009, 0.001],
                       [0.002, 0.008]])
        p1 = np.dot( R, dac )
        return p1

def to_plain(arr):
    if arr.ndim==1:
        return arr.tolist()
    elif arr.ndim==2:
        return [i.tolist() for i in arr]
    else:
        raise NotImplementedError('')

def main():
    fc = FakeCalibration()
    fname = sys.argv[1]

    daca, dacb = np.mgrid[0:0xFFFF:100j, 0:0xFFFF:100j]
    daca = daca.ravel().astype(np.uint16)
    dacb = dacb.ravel().astype(np.uint16)
    dac = np.array([daca,dacb])
    pixels = fc.pixels_for_dac( dac )
    print 'dac.shape',dac.shape
    print 'pixels.shape',pixels.shape
    save_raw_calibration_data(fname, dac, pixels)

if __name__=='__main__':
    main()
