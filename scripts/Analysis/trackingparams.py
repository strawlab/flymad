import numpy as np

import adskalman.adskalman

class Kalman:
    ### KEEP THESE IN SYNC WITH FLYMAD TRACKER

    FPS = 100
    Qsigma=10.0 # process covariance
    Rsigma=10.0 # observation covariance

    dt = 1.0/FPS

    # process model
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]],
                 dtype=np.float64)
    # observation model
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]],
                 dtype=np.float64)
    # process covariance
    Q = Qsigma*np.eye(4)
    # measurement covariance
    R = Rsigma*np.eye(2)

    def smooth(self, x, y):

        y = np.c_[x.values,y.values]
        initx = np.array([y[0,0],y[0,1],0,0])
        initV = 0*np.eye(4)

        xsmooth,Vsmooth = adskalman.adskalman.kalman_smoother(y,
                                self.A,self.C,
                                self.Q,self.R,
                                initx,initV)

        return xsmooth

