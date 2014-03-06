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

class NoCalibration(Exception):
    pass

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

        if not calibs:
            raise NoCalibration("No calibration detected in %s" % fname)

        #remove identical calib strings
        calibs = set(calibs)
        if len(calibs) != 1:
            raise ValueError("Multiple different calibrations detected in same bag file")

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

def load_calibration(fname, **kwargs):
    dac, pixels = read_raw_calibration_data(fname)
    cal = Calibration(dac, pixels, **kwargs)
    return cal

class Calibration:

    def __init__(self, dac, pixels, verbose=False):
        self.dac = dac
        self.pixels = pixels
        self._verbose = verbose

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

        self.d2px = LinearNDInterpolator(dac.T, pixels[0,:])
        self.d2py = LinearNDInterpolator(dac.T, pixels[1,:])

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

        self._mean_da_error = np.mean(da_errors)
        self._mean_db_error = np.mean(db_errors)

        #print 'Mean reprojection errors: %.1f, %.1f (DACa, DACb)'%(mean_da_error,mean_db_error)

    def __getstate__(self):
        d = self.__dict__.copy()
        #LinearNDInterpolator cannot be pickled in versions of scipy <= 0.11
        #https://github.com/scipy/scipy/pull/172
        d.pop('d2px')
        d.pop('d2py')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.d2px = LinearNDInterpolator(self.dac.T, self.pixels[0,:])
        self.d2py = LinearNDInterpolator(self.dac.T, self.pixels[1,:])

    def __repr__(self):
        if self._verbose:
            da, db = self.dac[:,0]
            # expected
            pxe, pye = self.pixels[:,0]
            d = np.c_[da,db]
            # actual
            pxa = self.d2px( d )[0]
            pya = self.d2py( d )[0]

            extra = ' (da:%s db:%s' % (da, db)
            extra += ' pxe:%s pye:%s' % (pxe, pye)
            extra += ' pxa:%s pya:%s)' % (pxa, pya)
        else:
            extra = ''

        return "<Calibration reproj_errors DACa:%.1f DACb:%.1f%s>" % (
                    self._mean_da_error, self._mean_db_error, extra)

    def __eq__(self, other):
        return np.allclose(self.dac,other.dac) and np.allclose(self.pixels,other.pixels)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_arena_measurements(self):
        """ returns the bounds of the arena

        (cx,cy,radius),xlim,ylim

        where xlim = (xmin,xmax)

        """
        xpx = self.pixels[0,:]
        ypx = self.pixels[1,:]
        xlim = xpx.min(),xpx.max()
        ylim = ypx.min(),ypx.max()

        try:
            from flymad.smallest_enclosing_circle import make_circle
            print "finding smallest enclosing circle of calibration camera pixels"
            circ = make_circle(self.pixels.T)
        except ImportError:
            #assume the circle is centered in xlim and ylim
            rx = (xlim[1]-xlim[0])/2.0
            ry = (ylim[1]-ylim[0])/2.0
            cr = max(rx,ry)
            cx = xlim[0] + rx
            cy = ylim[0] + ry
            circ = (cx, cy, cr)

        return map(int,circ), map(int,xlim), map(int,ylim)

