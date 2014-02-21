import pandas as pd
import sys, os,re
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import collections

import madplot

# frame: 149112
# time: 02:34:18
# roughly 1392687253.9

from th_experiments import DOROTHEA_NAME_RE_BASE, DOROTHEA_BAGDIR
DOROTHEA_NAME_REGEXP = re.compile(r'^' + DOROTHEA_NAME_RE_BASE + '.mp4.csv$')
BAG_DATE_FMT = "%Y-%m-%d-%H-%M-%S.bag"

Scored = collections.namedtuple('Scored', 'bag csv')

def load_bagfile_get_laseron(arena, score, smooth):
    NO_PROBOSCIS = 0
    PROBOSCIS = 1

    l_df, t_df, h_df, geom = madplot.load_bagfile(score.bag, arena, smooth=smooth)

    scored_ix = [t_df.index[0]]
    scored_v = [NO_PROBOSCIS]

    for idx,row in pd.read_csv(score.csv).iterrows():
        csv_row_frame = row['framenumber']
        if np.isnan(csv_row_frame):
            continue
        matching = t_df[t_df['t_framenumber'] == int(csv_row_frame)]
        if len(matching) == 1:
            scored_ix.append(matching.index[0])
            rowval = row['as']
            if rowval=='a':
                scored_v.append( PROBOSCIS )
            elif rowval=='s':
                scored_v.append( NO_PROBOSCIS )
            elif np.isnan(rowval):
                scored_v.append(np.nan)
            else:
                raise ValueError('unknown row value: %r'%rowval)

    s = pd.Series(scored_v,index=scored_ix)
    t_df['proboscis'] = s
    t_df['proboscis'].fillna(method='ffill', inplace=True)
#    t_df['Vfwd'] = t_df['v'] * t_df['fwd']

    lon = l_df[l_df['laser_power'] > 0]

    if len(lon):
        ton = lon.index[0]

        l_off_df = l_df[ton:]
        loff = l_off_df[l_off_df['laser_power'] == 0]
        toff = loff.index[0]

        onoffset = np.where(t_df.index >= ton)[0][0]
        offoffset = np.where(t_df.index >= toff)[0][0]

        return t_df, onoffset

    else:
        return None, None

def get_bagfilename( bag_dirname, dtval ):
    MP4_DATE_FMT = "%Y%m%d_%H%M%S"
    mp4time = time.strptime(dtval, MP4_DATE_FMT)

    inputbags = glob.glob( os.path.join(bag_dirname,'*.bag') )
    best_diff = np.inf
    bname = None
    for bag in inputbags:
        bagtime = time.strptime(os.path.basename(bag), BAG_DATE_FMT)
        this_diff = abs(time.mktime(bagtime)-time.mktime(mp4time))
        if this_diff < best_diff:
            bname = bag
            best_diff = this_diff
        else:
            continue
    if best_diff > 10.0:
        raise RuntimeError( 'no bagfile found for %s'%dtval)
    assert os.path.exists(bname)
    return bname


def my_subplot( n_conditions ):
    if n_conditions==3:
        n_rows, n_cols = 2,2
    elif n_conditions==4:
        n_rows, n_cols = 2,2
    elif n_conditions==5:
        n_rows, n_cols = 3,2
    elif n_conditions==6:
        n_rows, n_cols = 3,2
    else:
        print 'n_conditions',n_conditions
        1/0
    return n_rows, n_cols

dirname = sys.argv[1]
if 1:
    dfs = collections.defaultdict(list)
    csv_files = glob.glob( os.path.join(dirname,'*.csv') )
    bag_dirname = DOROTHEA_BAGDIR #os.path.abspath(os.path.join(dirname,'..','..','TH_Gal4_bagfiles'))
    for csv_filename in csv_files:
        matchobj = DOROTHEA_NAME_REGEXP.match(os.path.basename(csv_filename))
        if matchobj is None:
            print "error: incorrectly named file?", csv_filename
            continue

        parsed_data = matchobj.groupdict()
        #print 'parsed_data',parsed_data
        #print 'CSV datetime',parsed_data['datetime']
        bag_filename = get_bagfilename( bag_dirname, parsed_data['datetime'] )
        arena = madplot.Arena('mm')
        smooth=True
#        l_df, t_df, h_df, geom = madplot.load_bagfile(bag_filename, arena, smooth=smooth)
        score = Scored(bag_filename, csv_filename)
        t_df,lon = load_bagfile_get_laseron(arena, score, smooth)
        assert t_df is not None

        # y = dataframe['as'].values
        # print y.dtype
        # x = np.arange( len(y) )
        # plt.plot(x,y,label='%s'%(parsed_data['condition'],))
        dfs[parsed_data['condition']].append( (t_df, parsed_data) )
    conditions = dfs.keys()
    n_conditions = len(conditions)
    fig = plt.figure()
    n_rows, n_cols = my_subplot( n_conditions )
    for i, condition in enumerate(conditions):
        ax = fig.add_subplot(n_rows, n_cols, i+1 )
        arrs = []
        for (t_df,parsed_data) in dfs[condition]:
            ax.plot(t_df['proboscis'],
                    label='%s (fly %s, trial %s)'%(parsed_data['condition'],
                                                   parsed_data['condition_flynum'],
                                                   parsed_data['trialnum'],
                                                   ))
            arrs.append( t_df['proboscis'] )
        all_arrs = np.array( arrs )
        mean_arr = np.mean( all_arrs, axis=0 )
        ax.plot( mean_arr, color='k', lw=2 )
        ax.legend()
    plt.show()
