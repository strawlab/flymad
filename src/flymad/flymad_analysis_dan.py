import math
import numpy as np

SECOND_TO_NANOSEC = 1e9

from trackingparams import Kalman

def kalman_smooth_dataframe(df):
    #we need dt in seconds to calculate velocity. numpy returns nanoseconds here
    #because this is an array of type datetime64[ns] and I guess it retains the
    #nano part when casting
    dt = np.gradient(df.index.values.astype('float64')/SECOND_TO_NANOSEC)

    #smooth the positions, and recalculate the velocitys based on this.
    kf = Kalman()
    smoothed = kf.smooth(df['x'].values, df['y'].values)
    _x = smoothed[:,0]
    _y = smoothed[:,1]
    _vx = np.gradient(_x) / dt
    _vy = np.gradient(_y) / dt
    _v = np.sqrt( (_vx**2) + (_vy**2) )

    df['x'] = _x
    df['y'] = _y
    df['vx'] = _vx
    df['vy'] = _vy
    df['v'] = _v

    return dt

def fix_scoring_colums(df, valmap={'zx':{'z':math.pi,'x':0},
                                   'as':{'a':1,'s':0}}):
    for col in valmap:
        for val,replace in valmap[col].items():
            df[col][df[col] == val] = replace

    for col in valmap:
        df[col] = df[col].astype(np.float64)

def fixup_index_and_resample(df, t):
    #tracked_t was the floating point timestamp, which when roundtripped through csv
    #could lose precision. The index is guarenteed to be unique, so recreate tracked_t
    #from the index (which is seconds since epoch)
    tracked_t = df.index.values.astype(np.float64) / SECOND_TO_NANOSEC
    df['tracked_t'] = tracked_t
    #but, you should not need to use tracked_t now anyway, because this dataframe
    #has a datetime index...
    #
    #YAY pandas
    df['time'] = df.index.values.astype('datetime64[ns]')
    df.set_index(['time'], inplace=True)
    #
    #now resample to 10ms (mean)
    df = df.resample(t)

    return df


