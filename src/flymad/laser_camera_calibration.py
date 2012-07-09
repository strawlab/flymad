import numpy as np
from scipy.interpolate import griddata
import json
import yaml

PIXEL_WIDTH=1024
PIXEL_HEIGHT=768

def load_calibration(fname):
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
    cal = Calibration( np.array(data['dac']), pixels )
    return cal

class Calibration:
    def __init__(self, dac, pixels):
        self.dac = dac
        self.pixels = pixels

        px, py = np.mgrid[0:PIXEL_HEIGHT, 0:PIXEL_WIDTH]

        try:
            # p2d
            self.p2da = griddata( pixels.T, dac[0,:], (px,py),
                                  method='cubic')
        except RuntimeError:
            self.p2da = None
        try:
            self.p2db = griddata( pixels.T, dac[1,:], (px,py),
                                  method='cubic')
        except RuntimeError:
            self.p2db = None
            

def main():
    import sys
    import matplotlib.pyplot as plt
    fname = sys.argv[1]
    cal = load_calibration(fname)
    plt.figure()
    #plt.plot( cal.pixels[0,:], cal.pixels[1,:], 'b.' )
    plt.plot( cal.pixels[0,:], cal.pixels[1,:], 'b.-' )

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    if cal.p2da is not None:
        ax.imshow(cal.p2da)
    ax = fig.add_subplot(2,1,2)
    if cal.p2db is not None:
        ax.imshow(cal.p2db)

    plt.show()

if __name__=='__main__':
    main()
