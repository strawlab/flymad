import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator, LinearNDInterpolator
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

class Calibration:
    def __init__(self, dac, pixels):
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
