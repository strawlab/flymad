import math
import glob
import os.path
import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.patches

SECOND_TO_NANOSEC = 1e9

from trackingparams import Kalman

def get_arena_conf(calibration_file=None):
    #DEFAULTS FROM THE FIRST NMETH SUBMISSION - AS USED FOR DANS
    #FIRST EXPERIMENTS

    #from clicking on the outermost points to define the circlt
    l = {'y': 253.9102822580644, 'x': 256.27620967741939}
    r = {'y': 241.01108870967732, 'x': 624.82459677419365}
    t = {'y': 30.017137096774036, 'x': 438.70766129032268}
    b = {'y': 457.5332661290322, 'x': 429.49395161290329}

    w = r['x'] - l['x']
    h = b['y'] - t['y']

    x = l['x'] + (w/2.0) + 12
    y = t['y'] + (h/2.0)
    r = min(w,h)/2.0

    #NO DAN. NEVER REVERSE THE AXIS BY SETTING xlim[-1] < xlim[0]
    #self._xlim = [self._y+self._r+10,self._y-self._r-10]

    xlim = [y-r-10,y+r+10]
    ylim = [x-r-10,x+r+10]

    rw = 0.045
    sx = 0.045/160
    sy = 0.045/185

    conf = {'cx':x,'cy':y,'cr':r,'rw':rw,'sx':sx,'sy':sy}

    return {"jsonconf":conf, "calibration":calibration_file}


class Arena:

    CONVERT_OPTIONS = {
        False:None,
        "m":1.0,
        "cm":100.0,
        "mm":1000.0,
    }

    def __init__(self, convert, jsonconf=dict()):
        #from clicking on the outermost points to define the circlt
        l = {'y': 253.9102822580644, 'x': 256.27620967741939}
        r = {'y': 241.01108870967732, 'x': 624.82459677419365}
        t = {'y': 30.017137096774036, 'x': 438.70766129032268}
        b = {'y': 457.5332661290322, 'x': 429.49395161290329}

        w = r['x'] - l['x']
        h = b['y'] - t['y']
        self._x = l['x'] + (w/2.0) + 12
        self._y = t['y'] + (h/2.0)
        self._r = min(w,h)/2.0

        #NO DAN. NEVER REVERSE THE AXIS BY SETTING xlim[-1] < xlim[0]
        #self._xlim = [self._y+self._r+10,self._y-self._r-10]

        self._xlim = [self._y-self._r-10,self._y+self._r+10]
        self._ylim = [self._x-self._r-10,self._x+self._r+10]

        self._convert = convert
        self._convert_mult = self.CONVERT_OPTIONS[convert]

        self._rw = float(jsonconf.get('rw',0.045))   #radius in m
        self._sx = float(jsonconf.get('sx',0.045/160)) #scale factor px->m
        self._sy = float(jsonconf.get('sy',0.045/185)) #scale factor px->m

        #cache the simgear object for quick tests if the fly is in the area
        (sgcx,sgcy),sgr = self.circ
        self._sg_circ = sg.Point(sgcx,sgcy).buffer(sgr)

    def __repr__(self):
        return "<ArenaDan cx:%.1f cy:%.1f r:%.1f sx:%f sy:%f>" % (
                    self._x,self._y,self._r,self._sx,self._sy)

    @property
    def unit(self):
        if self._convert:
            return self._convert
        else:
            return 'px'
    @property
    def circ(self):
        if self._convert:
            return (0,0), self._convert_mult*self._rw
        return (self._x,self._y), self._r
    @property
    def cx(self):
        return self.scale_x(self._x)
    @property
    def cy(self):
        return self.scale_y(self._y)
    @property
    def r(self):
        if self._convert:
            return self._convert_mult*self._rw
        return self._r
    @property
    def xlim(self):
        if self._convert:
            #10% larger than the radius
            xlim = np.array([-self._rw, self._rw])*self._convert_mult
            return (xlim + (xlim*0.1)).tolist()
        return self._xlim
    @property
    def ylim(self):
        if self._convert:
            #10% larger than the radius
            ylim = np.array([-self._rw, self._rw])*self._convert_mult
            return (ylim + (ylim*0.1)).tolist()
        return self._ylim

    def scale_x(self, x, origin=None):
        if self._convert:
            if origin is not None:
                o = origin
            else:
                o = self._x
            if isinstance(x,list):
                x = np.array(x)
            return (x-o)*self._sx*self._convert_mult
        return x

    def scale_y(self, y, origin=None):
        if self._convert:
            if origin is not None:
                o = origin
            else:
                o = self._y
            if isinstance(y,list):
                y = np.array(y)
            return (y-o)*self._sy*self._convert_mult
        return y

    def scale_vx(self, x):
        #when scaling velocity, don't adjust for the origin
        return self.scale_x(x, origin=0.0)

    def scale_vy(self, y):
        #when scaling velocity, don't adjust for the origin
        return self.scale_y(y, origin=0.0)

    def scale(self, v):
        if self._convert:
            if self._sx == self._sy:
                s = self._sx
            else:
                print "warning: x and y scale not identical"
                s = (self._sx + self._sy) / 2.0
            if isinstance(v,list):
                v = np.array(v)
            return v*s*self._convert_mult
        else:
            return v

    def get_intersect_polygon(self, geom):
        if self._convert and geom:
            points_x, points_y = geom
            geom = (map(self.scale_x,points_x),map(self.scale_y,points_y))

        if geom:
            poly = sg.Polygon(list(zip(*geom)))
            inter = self._sg_circ.intersection(poly)
            return inter
        else:
            return None

    def get_intersect_points(self, geom):
        inter = self.get_intersect_polygon(geom)
        if inter:
            return list(inter.exterior.coords)
        else:
            return []

    def get_intersect_patch(self, geom, **kwargs):
        pts = self.get_intersect_points(geom)
        if pts:
            return matplotlib.patches.Polygon(pts, **kwargs)
        return None

    def get_patch(self, **kwargs):
        (cx,cy),r = self.circ
        return matplotlib.patches.Circle((cx,cy), radius=r, **kwargs)

    def get_limits(self):
        #(xlim, ylim)
        return self.xlim, self.ylim

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
        

def kalman_smooth_dataframe(df, arena=None, smooth=True):
    if arena:
        fsx = arena.scale_x
        fsy = arena.scale_y
    else:
        fsx = fsy = lambda _v: _v

    #we need dt in seconds to calculate velocity. numpy returns nanoseconds here
    #because this is an array of type datetime64[ns] and I guess it retains the
    #nano part when casting
    dt = np.gradient(df.index.values.astype('float64')/SECOND_TO_NANOSEC)

    if smooth:
        print "smoothing"
        #smooth the positions, and recalculate the velocitys based on this.
        kf = Kalman()
        smoothed = kf.smooth(df['x'].values, df['y'].values)
        _x = fsx(smoothed[:,0])
        _y = fsy(smoothed[:,1])
    else:
        _x = fsx(df['x'].values)
        _y = fsy(df['y'].values)

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


