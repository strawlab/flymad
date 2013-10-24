import math
import glob
import os.path
import numpy as np
import pandas as pd

SECOND_TO_NANOSEC = 1e9

from trackingparams import Kalman

def courtship_combine_csvs_to_dataframe(path, globpattern=None):
    filelist = []

    if globpattern is None:
        globpattern = '*.csv'

    globpattern = os.path.join(path,globpattern)

    for posfile in sorted(glob.glob(globpattern)):
        csvfilefn = os.path.basename(posfile)
        df = pd.read_csv(posfile)
        try:
            experimentID,date,time = csvfilefn.split("_",2)
            genotype,laser,repID = experimentID.split("-",2)
            repID = experimentID + "_" + date
        except:
            print "invalid filename:", csvfilefn
            continue 
        #CONCATENATE MATCHING IDs:
        if csvfilefn in filelist:
            continue   #avoid processing flies more than once.
        filelist.append(csvfilefn)  
        print "processing:", csvfilefn         
        for csvfile2 in sorted(glob.glob(globpattern)):
            csvfile2fn = os.path.basename(csvfile2)
            try:
                experimentID2,date2,time2 = csvfile2fn.split("_",2)
                genotype2,laser2,repID2 = experimentID2.split("-",2)
                repID2 = experimentID2 + "_" + date2
            except:
                continue
            if csvfile2fn in filelist:
                continue 
            elif repID2 == repID:
                print "    concatenating:", csvfile2fn
                filelist.append(csvfile2fn)
                csv2df = pd.read_csv(csvfile2)
                csv2df = pd.DataFrame(csv2df)
                df = pd.concat([df, csv2df])
            else:
                continue
  
        #convert 'V', 'X' AND 'S' to 1 or 0
        df['zx'] = df['zx'].astype(object).fillna('x')
        df['as'] = df['as'].astype(object).fillna('s')
        df['cv'] = df['cv'].astype(object).fillna('v')
        df['as'].fillna(value='s')
        df['cv'].fillna(value='v')        
        df['zx'][df['zx'] == 'z'] = 1
        df['cv'][df['cv'] == 'c'] = 1
        df['as'][df['as'] == 'a'] = 1
        df['zx'][df['zx'] == 'x'] = 0
        df['cv'][df['cv'] == 'v'] = 0
        df['as'][df['as'] == 's'] = 0
        
        #MATCH COLUMN NAMES (OLD VS NEW flymad_score_movie)
        datenum = int(date)
        if datenum >= 20130827:
            df = df.drop('as',axis=1)
            df = df.rename(columns={'tracked_t':'t', 'laser_state':'as'}) #, inplace=False
            df['as'] = df['as'].fillna(value=0)
        else:
            pass           

        df[['t','theta','v','vx','vy','x','y','zx','as','cv']] = df[['t','theta','v','vx','vy','x','y','zx','as','cv']].astype(float)

        yield df, (csvfilefn,experimentID,date,time,genotype,laser,repID)
        

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


