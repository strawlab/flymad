# -*- coding: utf-8 -*-

import math
import glob
import os.path
import datetime
import calendar
import time
import re

import numpy as np
import pandas as pd
import shapely.geometry as sg
import matplotlib.patches
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

SECOND_TO_NANOSEC = 1e9

from trackingparams import Kalman

GENOTYPE_LABELS = {
    "wtrpmyc":"+/TRPA1","wtrp":"+/TRPA1",
    "G323":"P1/+","wGP":"P1/TRPA1",
    "40347":"pIP10/+","40347trpmyc":"pIP10/TRPA1",
    "5534":"vPR6/+","5534trpmyc":"vPR6/TRPA1",
    "43702":"vMS11/+","43702trp":"vMS11/TRPA1",
    "41688":"dPR1/+","41688trp":"dPR1/TRPA1",
}

HUMAN_LABELS = {
    "wtrpmyc":"UAS-control","wtrp":"UAS-control",
    "G323":"Gal4-control","G323_specific":"P1 Gal4-control","wGP":"P1>TRPA1",
    "40347":"Gal4-control","40347_specific":"pIP10 Gal4-control","40347trpmyc":"pIP10>TRPA1",
    "5534":"Gal4-control","5534trpmyc":"vPR6>TRPA1",
    "43702":"Gal4-control","43702trp":"vMS11>TRPA1",
    "41688":"Gal4-control","41688trp":"dPR1>TRPA1",
    "50660trp":"Moonwalker","50660":"Gal4-control",
}

UNIT_SPACE = u"\u205F"
UNIT_SPACE = " "

class AlignError(Exception):
    pass

def laser_power_string(wavelength, current):
    if wavelength == 808:
        #Optical Power (mW) = 597.99(current in A) - 51.78
        mw = 597.99*float(current) - 51.78
        return u"%.0f%smW" % (mw, UNIT_SPACE)
    elif wavelength == 635:
        #Optical Power (uW) = 59278(current in A) - 1615
        uw = 59278*float(current) - 1615
        return u"%.0f%sµW" % (uw, UNIT_SPACE)
    else:
        return u"%.0f%smA" % current

def laser_wavelength_string(wavelength):
    return u"%d%snm" % (int(wavelength), UNIT_SPACE)

def laser_desc(specifier=None, wavelength=None, current=None):

    def _laser_desc(w,c):
        return "%s %s" % (laser_wavelength_string(w),
                          laser_power_string(w,c))

    #in dans experiments, specifier is
    #350iru, 350ru
    #130ht,120h,120t,140hpc
    #what consistency!!!
    if not any((specifier, wavelength, current)):
        raise Exception("You must provide specifier, wavelength or current")

    if all((wavelength, current)):
        return _laser_desc(wavelength, current)
    else:
        label = specifier

        try:
            match = re.match('([0-9]{2,})([iru]{2,})', specifier)
            match_ir = re.match('([0-9]{2,})([htpc]{1,})', specifier)
            if match:
                current_ma,wavelength = match.groups()
                wavelength = {'iru':808,'ru':635}[wavelength]
                if (int(current_ma) == 350) and (int(wavelength) == 635):
                    #fix dan mislabeling 35ma red laser as 350ma
                    current_ma = 35
                label = _laser_desc(wavelength, float(current_ma)*1e-3)
            elif match_ir:
                current_ma,target = match_ir.groups()
                label = _laser_desc(808, float(current_ma)*1e-3)
        except (KeyError, ValueError, TypeError), e:
            print "WARNING: Error parsing",specifier
            label += "!!!"

    return label

def cmp_laser_desc(desc_a, desc_b):
    UNIT_MULT = {"mA":1,"mW":1e-3,u"uW":1e-6}

    def _get_cval(desc):
        wl,_,pwr,pwr_unit = desc.split(' ')
        #I could not get dicts with unicode keys to work...
        if pwr_unit == u"µW":
            mult = 1e-6
        elif pwr_unit == "mW":
            mult = 1e-3
        else:
            mult = 1.0
        pwr = float(pwr) * mult
        return float(wl) + pwr
    return cmp(_get_cval(desc_a), _get_cval(desc_b))

def cmp_laser(laser_a, laser_b):
    return cmp_laser_desc(laser_desc(laser_a), laser_desc(laser_b))

def human_label(gt,specific=False):
    gts = gt+"_specific" if specific else gt
    return HUMAN_LABELS.get(gts,HUMAN_LABELS.get(gt,gt))

def genotype_label(gt):
    return GENOTYPE_LABELS.get(gt,gt)

def genotype_is_exp(gt):
    return re.match('\w+/\w+$', genotype_label(gt)) is not None

def genotype_is_ctrl(gt):
    return re.match('\w+/\+$', genotype_label(gt)) is not None

def genotype_is_trp_ctrl(gt):
    return re.match('\+/\w+$', genotype_label(gt)) is not None

def get_genotype_order(gt):
    if genotype_is_exp(gt):
        order = 1 if gt == 'wGP' else 2
    elif genotype_is_ctrl(gt):
        order = 3
    else:
        order = 4
    return order

def to_si(d,space=''):
    incPrefixes = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    decPrefixes = ['m', 'µ', 'n', 'p', 'f', 'a', 'z', 'y']

    degree = int(math.floor(math.log10(math.fabs(d)) / 3))

    prefix = ''

    if degree != 0:
        ds = degree/math.fabs(degree)
        if ds == 1:
            if degree - 1 < len(incPrefixes):
                prefix = incPrefixes[degree - 1]
            else:
                prefix = incPrefixes[-1]
                degree = len(incPrefixes)
        elif ds == -1:
            if -degree - 1 < len(decPrefixes):
                prefix = decPrefixes[-degree - 1]
            else:
                prefix = decPrefixes[-1]
                degree = -len(decPrefixes)

        scaled = float(d * math.pow(1000, -degree))
        s = "{scaled}{space}{prefix}".format(scaled=scaled,space=space,prefix=prefix)

    else:
        s = "{d}".format(d=d)

    return s


def scoring_video_mp4_click(image_path):
    img = mimg.imread(image_path)
    fig1 = plt.figure()
    fig1.set_size_inches(12,8)
    fig1.subplots_adjust(hspace=0)
    ax1 = fig1.add_subplot(1,1,1)

    #the original wide field camera was 659x494px. The rendered mp4 is 384px high
    #the widefield image is padded with a 10px margin, so it is technically 514 high.
    #scaling h=384->514 means new w=1371
    #
    #the image origin is top-left because matplotlib
    ax1.imshow(img, extent=[0,1371,514,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
    ax1.axis('off')

    targets = []
    def _onclick(target):
        #subtract 10px for the margin
        xydict = {'x': target.xdata-10, 'y': target.ydata-10}
        targets.append(xydict)

    cid = fig1.canvas.mpl_connect('button_press_event', _onclick)
    plt.show()
    fig1.canvas.mpl_disconnect(cid)

    return targets


def get_arena_conf(calibration_file=None, **kwargs):
    #DEFAULTS FROM THE FIRST NMETH SUBMISSION - AS USED FOR DANS
    #FIRST EXPERIMENTS

    #from clicking on the outermost points to define the circlt
    l = kwargs.get('l',{'y': 253.9102822580644, 'x': 256.27620967741939})
    r = kwargs.get('r',{'y': 241.01108870967732, 'x': 624.82459677419365})
    t = kwargs.get('t',{'y': 30.017137096774036, 'x': 438.70766129032268})
    b = kwargs.get('b',{'y': 457.5332661290322, 'x': 429.49395161290329})

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

    if 'l' not in kwargs:
        #keep backwards compat with dan's first nmeth submission
        sx = 0.045/160
        sy = 0.045/185
    else:
        #the more correct case for a camera viewing the arena
        #perpendicuarly, and the user supplying l,r,t,b points
        sx = 0.045/(w/2.0)
        sy = 0.045/(h/2.0)

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

def create_object_id(datestr=None, timestr=None):
    #date = 20140216
    #time = 174913.mp4.csv

    ts = None
    if datestr and timestr:
        try:
            dt = datetime.datetime.strptime("%s_%s" % (datestr,timestr.split('.')[0]),
                                           "%Y%m%d_%H%M%S")
            ts = calendar.timegm(dt.timetuple())
        except ValueError:
            print "invalid date/time", datestr, timestr

    if ts is None:
        ts = int(time.time()*1e9)

    return ts

def courtship_combine_csvs_to_dataframe(path, globpattern=None, as_is_laser_state=True):
    filelist = []

    if globpattern is None:
        globpattern = '*.csv'

    globpattern = os.path.join(path,globpattern)

    for obj_id,posfile in enumerate(sorted(glob.glob(globpattern))):
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
                if 'rescore' in csvfile2fn:
                    print "    rescore:", csvfile2fn, " replaces ", csvfilefn
                    continue
                else:
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

        #give files a semi-random obj_id
        df['obj_id'] = create_object_id(date,time)

        cols = ['t','theta','v','vx','vy','x','y','zx','as','cv', 'obj_id']
        if  as_is_laser_state:
            #backwards compatibility...
            #MATCH COLUMN NAMES (OLD VS NEW flymad_score_movie)
            datenum = int(date)
            if datenum >= 20130827:
                df = df.drop('as',axis=1)
                df = df.rename(columns={'tracked_t':'t', 'laser_state':'as'}) #, inplace=False
                df['as'] = df['as'].fillna(value=0)

        else:
            df = df.rename(columns={'tracked_t':'t'})
            cols.append('laser_state')

        try:
            df[cols] = df[cols].astype(float)
        except KeyError, e:
            print "INVALID DATAFRAME %s (%s)\n\thas: %s" % (posfile, e, ','.join(df.columns))
            continue

        #git the dateframe a proper datetime index for resampling
        #first remove rows with NaN t values
        df = df[np.isfinite(df['t'])]
        df.index = [datetime.datetime.fromtimestamp(t) for t in df['t']]

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
        print "\tsmoothing (%r)" % arena
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

def _resample_freq(resample_specifier):
    MULTIPLIER = {'L':1000.0,'S':1.0}
    m = MULTIPLIER[resample_specifier[-1]]
    v = float(resample_specifier[:-1])
    return m/v

def  _resample_dt(resample_specifier):
    return 1.0 / _resample_freq(resample_specifier)

def get_num_rows(desired_seconds, resample_specifier='10L'):
    return desired_seconds * _resample_freq(resample_specifier)

def get_resampled_timebase(desired_seconds, resample_specifier='10L'):
    n = get_num_rows(desired_seconds, resample_specifier)
    return np.linspace(0, desired_seconds, n, endpoint=False)

def get_resampled_timebase_from_df(df, resample_specifier='10L'):
    #the dataframe has been resampled to X, therefor its true lenght
    #in real seconds is 
    f = _resample_freq(resample_specifier)
    len_s = len(df) / f
    return np.arange(0, len_s, 1.0/f)

def align_t_by_laser_on(df, min_experiment_duration, align_first_only, t_range=None, min_num_ranges=1, exact_num_ranges=None):

    duration = (df.index[-1] - df.index[0]).total_seconds()
    if duration < min_experiment_duration:
        raise AlignError("dataframe too short (%ss vs %ss)" % (duration, min_experiment_duration))

    if align_first_only:
        min_num_ranges = 1

    #Here we have a 10ms resampled dataframe at least min_experiment_duration seconds long.
    df = df.head(get_num_rows(min_experiment_duration))

    #find when the laser first came on (argmax returns the first true value if
    #all values are identical
    #
    #gradient on an array of 0/1 using a distance of 1 (default) uses
    #one sample each side, giving 2x the number of non-zero values.
    dlaser = np.gradient( (df['laser_state'].values > 0).astype(int) ) > 0
    #but if we treat the array as a string we can find those edges \x00\01
    dlaser_str = dlaser.tostring()
    rising_edges = [n for n in xrange(len(dlaser_str)) if dlaser_str.find('\x00\x01', n) == n]

    n_rising_edges = len(rising_edges)
    print "\t%s laser pulses" % n_rising_edges

    if n_rising_edges < min_num_ranges:
        raise AlignError('insufficient number of laser on periods (%s)' % n_rising_edges)

    if (exact_num_ranges is not None) and (n_rising_edges != exact_num_ranges):
        raise AlignError('incorrect umber of laser on periods (%s vs %s). delete your cache' % (n_rising_edges,exact_num_ranges))

    if align_first_only:
        tb = get_resampled_timebase(min_experiment_duration)
        t0 = tb[rising_edges[0]]
        df['t'] = tb - t0
        df['trial'] = 1.0
        return df

    t_before, t_after = t_range
    ranges = []
    for edge_idx in rising_edges:
        t = df.index[edge_idx]
        ranges.append((t+DateOffset(seconds=t_before),t+DateOffset(seconds=t_after)))

    df['t'] = np.nan
    df['trial'] = np.nan
    for range_number,r in enumerate(ranges):
        r_df = df[r[0]:r[1]]
        r_duration = (r_df.index[-1] - r_df.index[0]).total_seconds()
        if r_duration < (t_after - t_before):
            #this trial was too short
            continue

        r_tb = get_resampled_timebase_from_df(r_df) + t_before
        r_df['t'] = r_tb
        r_df['trial'] = float(range_number)

    return df

def fixup_index_and_resample(df, resample_specifier='10L'):
    if resample_specifier != '10L':
        print "WARNING: Your code may assume 10ms (100fps). Are you sure you want to resample to", resample_specifier

    #tracked_t was the floating point timestamp, which when roundtripped through csv
    #could lose precision. The index is guarenteed to be unique, so recreate tracked_t
    #from the index (which is seconds since epoch)
    tracked_t = df.index.values.astype(np.float64) / SECOND_TO_NANOSEC
    df['tracked_t'] = tracked_t
    df['tracked_tns'] = df.index.values
    #but, you should not need to use tracked_t now anyway, because this dataframe
    #has a datetime index...
    #
    #YAY pandas
    df['time'] = df.index.values.astype('datetime64[ns]')
    df.set_index(['time'], inplace=True)
    #
    #now resample to 10ms (mean)
    df = df.resample(resample_specifier, fill_method='ffill')

    return df

def extract_metadata_from_filename(csvfile):
    csvfilefn = os.path.basename(csvfile)
    try:
        experimentID,date,time = csvfilefn.split("_",2)
        genotype,laser,repID = experimentID.split("-",2)
        repID = repID + "_" + date
    except:
        return None

    return experimentID,date,time,genotype,laser,repID

def load_and_smooth_csv(csvfile, arena, smooth, resample_specifier='10L'):
    metadata = extract_metadata_from_filename(csvfile)
    if metadata is None:
        print "WARNING: invalid filename:", csvfile
        return None

    experimentID,date,time,genotype,laser,repID = metadata
    print "processing:", experimentID

    df = pd.read_csv(csvfile, index_col=0)

    if not df.index.is_unique:
        print "\tWARNING: corrupt csv: index (ns since epoch) must be unique"
        return None

    #remove rows before we have a position
    q = pd.isnull(df['x']).values
    first_valid_row = np.argmin(q)
    df = df.iloc[first_valid_row:]
    print "\tremove %d invalid rows at start of file" % first_valid_row

    #convert 'V', 'X' AND 'S' to 1 or 0
    df['zx'] = df['zx'].astype(object).fillna('x')
    df['as'] = df['as'].astype(object).fillna('s')
    df['cv'] = df['cv'].astype(object).fillna('v')
    df['zx'][df['zx'] == 'z'] = 1
    df['cv'][df['cv'] == 'c'] = 1
    df['as'][df['as'] == 'a'] = 1
    df['zx'][df['zx'] == 'x'] = 0
    df['cv'][df['cv'] == 'v'] = 0
    df['as'][df['as'] == 's'] = 0

    #ensure these are floats incase we later resample
    cols = ['zx','as','cv']
    df[cols] = df[cols].astype(float)

    #resample to 10ms (mean) and set a proper time index on the df
    df = fixup_index_and_resample(df, resample_specifier)

    #smooth the positions, and recalculate the velocitys based on this.
    dt = kalman_smooth_dataframe(df, arena, smooth)

    if 'laser_state' in df.columns:
        df['laser_state'] = df['laser_state'].fillna(value=0)
        #the resampling above, using the default rule of 'mean' will, if the laser
        #was on any time in that bin, increase the mean > 0.
        df['laser_state'][df['laser_state'] > 0] = 1

    return df,dt,experimentID,date,time,genotype,laser,repID

def load_courtship_csv(path):
    globpattern = os.path.join(path,"*.csv")
    for csvfile in sorted(glob.glob(globpattern)):
        csvfilefn = os.path.basename(csvfile)

        if 'rescore' in csvfilefn:
            print "ignoring:",csvfilefn
            continue

        metadata = extract_metadata_from_filename(csvfile)
        if metadata is None:
            print "WARNING: invalid filename:", csvfile
            continue

        experimentID,date,time,genotype,laser,repID = metadata
        print "processing:", csvfilefn

        df = pd.read_csv(csvfile, index_col=0)

        if not df.index.is_unique:
            print "\tWARNING: corrupt csv: index (ns since epoch) must be unique"
            continue

        #remove rows before we have a position
        q = pd.isnull(df['laser_state']).values
        first_valid_row = np.argmin(q)
        df = df.iloc[first_valid_row:]
        print "\tremove %d empty rows at start of file" % first_valid_row

        #make sure we always have a 't' column (for back compat)
        df['t'] = df.index.values / SECOND_TO_NANOSEC
        df['time'] = df.index.values.astype('datetime64[ns]')
        df.set_index(['time'], inplace=True)

        #convert 'V', 'X' AND 'S' to 1 or 0
        df['zx'] = df['zx'].astype(object).fillna('x')
        df['as'] = df['as'].astype(object).fillna('s')
        df['cv'] = df['cv'].astype(object).fillna('v')
        df['zx'][df['zx'] == 'z'] = 1
        df['cv'][df['cv'] == 'c'] = 1
        df['as'][df['as'] == 'a'] = 1
        df['zx'][df['zx'] == 'x'] = 0
        df['cv'][df['cv'] == 'v'] = 0
        df['as'][df['as'] == 's'] = 0

        #ensure these are floats incase we later resample
        cols = ['zx','as','cv']
        df[cols] = df[cols].astype(float)

        #give files a semi-random obj_id
        df['obj_id'] = create_object_id(date,time)

        yield df, (csvfilefn,experimentID,date,time,genotype,laser,repID)

if __name__ == "__main__":
    for x in ["350ru","033ru","434iru","183iru","130ht","120h","120t","140hpc"]:
        print x,"->",laser_desc(x)

    import sys
    print scoring_video_mp4_click(sys.argv[1])


