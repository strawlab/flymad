import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator
import json
import yaml

def to_plain(arr):
    """Convert numpy arrays to pure-Python types. Useful for saving to JSON"""
    if arr.ndim==1:
        return arr.tolist()
    elif arr.ndim==2:
        return [i.tolist() for i in arr]
    else:
        raise NotImplementedError('')

def save_raw_calibration_data(fname,dac,pixels):
    to_save = {'dac':to_plain(dac),
               'pixels':to_plain(pixels.astype(np.int))}
    fd = open(fname,mode='w')
    json.dump(to_save,fd) # JSON is valid YAML. And faster.
    fd.close()

def read_raw_calibration_data(fname):
    fd = open(fname,mode='r')
    try:
        # speculative attempt to open as JSON (a faster, and valid
        # subset of YAML)
        data = json.load(fd)
    except ValueError:
        fd.seek(0)
        data = yaml.load(fd)

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

def convert_dac(indac):
    # convert to actual linear scale used (two's complement)
    return np.array(indac).astype(np.int16)

class Calibration:
    def __init__(self, dac, pixels):
        dac = convert_dac(dac)
        self.dac = dac
        self.pixels = pixels
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
        self.d2px = CloughTocher2DInterpolator(dac.T, pixels[0,:])
        self.d2py = CloughTocher2DInterpolator(dac.T, pixels[1,:])

def main():
    import sys
    import matplotlib.pyplot as plt
    fname = sys.argv[1]
    cal = load_calibration(fname)

    if 0:
        plt.figure()
        plt.plot( cal.pixels[0,:], cal.pixels[1,:], 'b.-' )
        ax = plt.gca()
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

    if 0:
        plt.figure()
        plt.plot( cal.dac[0,:], cal.dac[1,:], 'b.-' )
        ax = plt.gca()
        ax.set_xlabel('a (DAC)')
        ax.set_ylabel('b (DAC)')

    if 1:
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        if cal.p2da is not None:
            ax.set_title('pixels->DACa')
            cax = ax.imshow(cal.p2da, origin='lower')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            fig.colorbar(cax)
        ax = fig.add_subplot(2,1,2)
        if cal.p2db is not None:
            ax.set_title('pixels->DACb')
            cax = ax.imshow(cal.p2db, origin='lower')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            fig.colorbar(cax)

    plt.show()

if __name__=='__main__':
    main()
