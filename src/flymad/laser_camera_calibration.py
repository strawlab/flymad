import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator, LinearNDInterpolator
import json
import yaml
import os.path

import roslib
import roslib.rosenv
import roslib.packages
roslib.load_manifest('flymad')
import rospy
import rosbag

def get_calibration_file_path(default=None, create=False, suffix='OUT.filtered'):
    if default and os.path.isfile(default):
        return default

    fn = "calibration%s.yaml" % suffix

    default = os.path.join(
                    roslib.rosenv.get_ros_home(),fn)
    #try ~/.ros/calibrationOUT.filtered.yaml first (for DAN backwards compat)
    if create or os.path.isfile(default):
        return default

    #always have a default for things like the viewer to work
    example = os.path.join(
                    roslib.packages.get_pkg_dir('flymad'),
                    'data','calibration','example',fn)

    return example

def to_plain(arr):
    """Convert numpy arrays to pure-Python types. Useful for saving to JSON"""
    if arr.ndim==1:
        return arr.tolist()
    elif arr.ndim==2:
        return [i.tolist() for i in arr]
    else:
        raise NotImplementedError('')

def save_raw_calibration_data(fname,dac,pixels):
    # convert pixels to integer values
    pixels_int = pixels.astype(np.int)
    bad_cond = np.isnan(pixels)
    pixels_int[bad_cond] = -9223372036854775808

    # create dictionary to save as JSON
    to_save = {'dac':to_plain(dac),
               'pixels':to_plain(pixels_int)}

    # save it
    fd = open(fname,mode='w')
    json.dump(to_save,fd) # JSON is valid YAML. And faster.
    fd.close()

def read_raw_calibration_data(fname):
    if fname.endswith('.yaml'):
        with open(fname, mode='r') as fd:
            data = yaml.load(fd)
    elif fname.endswith('.json'):
        with open(fname, mode='r') as fd:
            data = json.load(fd)
    elif fname.endswith('.bag'):
        calibs = []
        with rosbag.Bag(fname, mode='r') as bag:
            for topic,msg,rostime in bag.read_messages(topics=['/targeter/calibration']):
                if topic == '/targeter/calibration':
                    calibs.append(msg.data)

        #remove identical calib strings
        calibs = set(calibs)
        if len(calibs) != 1:
            raise Exception("Multiple different calibrations detected in same bag file")

        data = yaml.load(calibs.pop())
    else:
        raise Exception("Only calibrations stored in .yaml, .json or .bag files supported "\
                        "(not %s files)" % fname)

    pixels = np.array(data['pixels'])
    bad_cond = pixels==-9223372036854775808
    pixels = pixels.astype(np.float)
    pixels[bad_cond] = np.nan
    dac = np.array(data['dac'])
    return dac, pixels

def load_calibration(fname):
    dac, pixels = read_raw_calibration_data(fname)
    cal = Calibration( dac, pixels )
    return cal

class Calibration:
    def __init__(self, dac, pixels):
        self.dac = dac
        self.pixels = pixels

        good_cond = ~np.isnan(self.pixels)
        n_good_pixels = np.sum(good_cond)
        if n_good_pixels==0:
            raise ValueError('the calibration has zero valid pixels')
        pix_h = np.max(pixels[1,:])+1
        pix_w = np.max(pixels[0,:])+1

        self.py, self.px = np.mgrid[0:pix_h, 0:pix_w]

        self.db, self.da = np.mgrid[float(np.min(dac[1])):float(np.max(dac[1])):500j,
                                    float(np.min(dac[0])):float(np.max(dac[0])):500j]

        #method='nearest'
        method='cubic'

        try:
            # p2d
            self.p2da = griddata( pixels.T, dac[0,:], (self.px,self.py),
                                  method=method)
        except RuntimeError:
            self.p2da = None

        try:
            self.p2db = griddata( pixels.T, dac[1,:], (self.px,self.py),
                                  method=method)
        except RuntimeError:
            self.p2db = None

        # this is the same function underlying griddata
        if 0:
            self._d2px = CloughTocher2DInterpolator(dac.T, pixels[0,:])
            self._d2py = CloughTocher2DInterpolator(dac.T, pixels[1,:])
        else:
            self._d2px = LinearNDInterpolator(dac.T, pixels[0,:])
            self._d2py = LinearNDInterpolator(dac.T, pixels[1,:])
        self.d2px = self._d2px
        self.d2py = self._d2py

        if 1:
            da, db = dac[:,0]
            #da,db = -15096, 12357
            #da,db = 5691, 4862
            # expected
            pxe, pye = pixels[:,0]
            d = np.c_[da,db]
            # actual
            pxa = self._d2px( d )[0]
            pya = self._d2py( d )[0]
            #pxe, pye = self.d2px( d )
            print 'da, db',da, db
            print 'pxe, pye',pxe, pye
            print 'pxa, pya',pxa, pya

        # calculate reprojection errors
        n_samples = pixels.shape[1]
        da_errors = []
        db_errors = []
        for i in range(n_samples):
            (px,py) = pixels[:,i]

            da_estimated = self.p2da[ py, px ]
            db_estimated = self.p2db[ py, px ]
            da_actual, db_actual = dac[:,i]

            this_da_error = abs(da_estimated-da_actual)
            this_db_error = abs(db_estimated-db_actual)

            da_errors.append( this_da_error )
            db_errors.append( this_db_error )
        mean_da_error = np.mean(da_errors)
        mean_db_error = np.mean(db_errors)

        print 'Mean reprojection errors: %.1f, %.1f (DACa, DACb)'%(mean_da_error,mean_db_error)

