#!/usr/bin/env python
import strawlab_mpl.defaults as smd
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds
import matplotlib
import cPickle as pickle
import datetime

def setup_defaults():
    rcParams = matplotlib.rcParams

    rcParams['legend.numpoints'] = 1
    rcParams['legend.fontsize'] = 'medium' #same as axis
    rcParams['legend.frameon'] = False
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1

smd.setup_defaults()
setup_defaults()

import pandas as pd
import sys, os,re
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import namedtuple, OrderedDict, defaultdict
import argparse


import madplot

from th_experiments import DOROTHEA_NAME_RE_BASE
DOROTHEA_NAME_REGEXP = re.compile(r'^' + DOROTHEA_NAME_RE_BASE + '.mp4.csv$')
BAG_DATE_FMT = "%Y-%m-%d-%H-%M-%S.bag"

Scored = namedtuple('Scored', 'bag csv')

def load_bagfile_get_laseron(arena, score, smooth):
    NO_PROBOSCIS = 0
    PROBOSCIS = 1

    NO_WING_MOVEMENTS = 0
    WING_MOVEMENTS = 1

    NO_JUMPING = 0
    JUMPING = 1

    NO_ABDOMENING = 0
    ABDOMENING = 1

    geom, bag_results = madplot.load_bagfile(score.bag, arena, smooth=smooth)
    l_df = bag_results['targeted']
    t_df = bag_results['tracked']
    h_df = bag_results['ttm']

    proboscis_scored_ix = [t_df.index[0]]
    proboscis_scored_v = [NO_PROBOSCIS]

    wing_scored_ix = [t_df.index[0]]
    wing_scored_v = [NO_WING_MOVEMENTS]

    jump_scored_ix = [t_df.index[0]]
    jump_scored_v = [NO_JUMPING]

    abdomen_scored_ix = [t_df.index[0]]
    abdomen_scored_v = [NO_ABDOMENING]

    score_df = pd.read_csv(score.csv)

    for idx,row in score_df.iterrows():
        csv_row_frame = row['framenumber']
        if np.isnan(csv_row_frame):
            continue

        matching = t_df[t_df['t_framenumber'] == int(csv_row_frame)]
        if len(matching) ==1:

            # proboscis ---------------
            proboscis_scored_ix.append(matching.index[0])
            rowval = row['as']
            if rowval=='a':
                proboscis_scored_v.append( PROBOSCIS )
            elif rowval=='s':
                proboscis_scored_v.append( NO_PROBOSCIS )
            elif np.isnan(rowval):
                #proboscis_scored_v.append( NO_PROBOSCIS )
                proboscis_scored_v.append(np.nan)
            else:
                raise ValueError('unknown row value: %r'%rowval)


            # wing ---------------
            wing_scored_ix.append(matching.index[0])
            rowval = row['zx']
            if rowval=='z':
                wing_scored_v.append( WING_MOVEMENTS )
            elif rowval=='x':
                wing_scored_v.append( NO_WING_MOVEMENTS )
            elif np.isnan(rowval):
                #wing_scored_v.append( NO_WING_MOVEMENTS )
                wing_scored_v.append(np.nan)
            else:
                raise ValueError('unknown row value: %r'%rowval)

            # jump ---------------
            jump_scored_ix.append(matching.index[0])
            rowval = row['cv']
            if rowval=='c':
                jump_scored_v.append( JUMPING )
            elif rowval=='v':
                jump_scored_v.append( NO_JUMPING )
            elif np.isnan(rowval):
                #jump_scored_v.append( NO_JUMPING )
                jump_scored_v.append(np.nan)
            else:
                raise ValueError('unknown row value: %r'%rowval)

            # abdomen ---------------
            abdomen_scored_ix.append(matching.index[0])
            rowval = row['qw']
            if rowval=='q':
                abdomen_scored_v.append( ABDOMENING )
            elif rowval=='w':
                abdomen_scored_v.append( NO_ABDOMENING )
            elif np.isnan(rowval):
                #abdomen_scored_v.append( NO_ABDOMENING )
                abdomen_scored_v.append(np.nan)
            else:
                raise ValueError('unknown row value: %r'%rowval)
        elif len(matching)==0:
            continue
        else:
            print "len(matching)",len(matching)
            # why would we ever get here?
            1/0

    s = pd.Series(proboscis_scored_v,index=proboscis_scored_ix)
#    if len(s.index) > 1000:
#        raise RuntimeError('file %r seems corrupt. Delete and re-score it.'%score.csv)
    t_df['proboscis'] = s

    s = pd.Series(wing_scored_v,index=wing_scored_ix)
    t_df['wing'] = s

    s = pd.Series(jump_scored_v,index=jump_scored_ix)
    t_df['jump'] = s

    s = pd.Series(abdomen_scored_v,index=abdomen_scored_ix)
    t_df['abdomen'] = s

    t_df['proboscis'].fillna(method='ffill', inplace=True)
    t_df['wing'].fillna(method='ffill', inplace=True)
    t_df['jump'].fillna(method='ffill', inplace=True)
    t_df['abdomen'].fillna(method='ffill', inplace=True)
#    t_df['Vfwd'] = t_df['v'] * t_df['fwd']

    lon = l_df[l_df['laser_power'] > 0]

    if len(lon):
        laser_on = lon.index[0]

        # l_off_df = l_df[laser_on:]
        # loff = l_off_df[l_off_df['laser_power'] == 0]
        # toff = loff.index[0]

        # onoffset = np.where(t_df.index >= laser_on)[0][0]
        # offoffset = np.where(t_df.index >= toff)[0][0]

        return t_df, laser_on

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
    if n_conditions==0:
        n_rows, n_cols = 1,1
    elif n_conditions==1:
        n_rows, n_cols = 1,1
    elif n_conditions==2:
        n_rows, n_cols = 2,1
    elif n_conditions==3:
        n_rows, n_cols = 2,2
    elif n_conditions==4:
        n_rows, n_cols = 2,2
    elif n_conditions==5:
        n_rows, n_cols = 3,2
    elif n_conditions==6:
        n_rows, n_cols = 3,2
    elif n_conditions==7:
        n_rows, n_cols = 3,3
    elif n_conditions==8:
        n_rows, n_cols = 3,3
    elif n_conditions==9:
        n_rows, n_cols = 3,3
    else:
        print 'n_conditions',n_conditions
        1/0
    return n_rows, n_cols

NAMES = OrderedDict([('th1stim_head',('head','TH>trpA1')),
                     ('th1stim_thorax',('thorax','TH>trpA1')),
                     ('thcstrpa11stim_head',('head','trpA1')),
                     ('thcstrpa11stim_thorax',('thorax','trpA1')),
                     ('thgal41stim_head',('head','TH')),
                     ('thgal41stim_thorax',('thorax','TH')),
                     ])

def make_cache_fname( arena, dirname, bag_dirname, smooth ):
    d = os.path.basename(dirname)
    b = os.path.basename(bag_dirname)
    return 'cache_%s_%s_%s_%s.pkl'%(arena.unit,d,b,smooth)

def prepare_data( arena, dirname, bag_dirname, smooth ):
    dfs = defaultdict(list)
    dirname = os.path.abspath(dirname)
    csv_files = glob.glob( os.path.join(dirname,'csvs','*.csv') )
    if len(csv_files)==0:
        raise ValueError('%r matched no .csv files'%csv_files)
    for csv_filename in csv_files:
        matchobj = DOROTHEA_NAME_REGEXP.match(os.path.basename(csv_filename))
        if matchobj is None:
            print "error: incorrectly named file?", csv_filename
            continue

        #print csv_filename
        parsed_data = matchobj.groupdict()
        parsed_data['condition'] = parsed_data['condition'].lower()
        #print parsed_data
        # if int(parsed_data['trialnum'])!=2:
        #     print "parsed_data['trialnum']",parsed_data['trialnum']
        #     continue
        #print 'parsed_data',parsed_data
        #print 'CSV datetime',parsed_data['datetime']
        try:
            bag_filename = get_bagfilename( bag_dirname, parsed_data['datetime'] )
        except:
            print 'FAILED TO FIND BAG FOR CSV',csv_filename
            raise



#        l_df, t_df, h_df, geom = madplot.load_bagfile(bag_filename, arena, smooth=smooth)
        score = Scored(bag_filename, csv_filename)
        try:
            t_df,laser_on = load_bagfile_get_laseron(arena, score, smooth)
        except:
            print 'FAILED TO GET LASER DATA FOR CSV',csv_filename
            raise
        assert t_df is not None

        # y = dataframe['as'].values
        # print y.dtype
        # x = np.arange( len(y) )
        # plt.plot(x,y,label='%s'%(parsed_data['condition'],))
        dfs[parsed_data['condition']].append( (t_df, parsed_data, laser_on) )


    cache_fname = make_cache_fname( arena, dirname, bag_dirname, smooth )
    print 'saveing cache',cache_fname
    pickle.dump(dfs, open(cache_fname,'wb'), -1)
    return dfs

def plot_data(arena, dirname, smooth, dfs):
    conditions = dfs.keys()
    n_conditions = len(conditions)

    DO_LATENCY_FIGURE = False
    DO_POSITION_FIGURE = True

    if DO_POSITION_FIGURE:
        fig = plt.figure()
        max_trials = 0
        for i, condition in enumerate(conditions):
            n_trials = len(dfs[condition])
            max_trials = max( max_trials, n_trials )

        ax = None
        for i, condition in enumerate(conditions):
            for j in range(max_trials):
                if j>=len(dfs[condition]):
                    break
                ax = fig.add_subplot(n_conditions, max_trials, i*max_trials+j+1,
                                     sharex=ax, sharey=ax)
                (t_df,parsed_data,laser_on) = dfs[condition][j]
                post_laser_df = t_df[ t_df.index > (laser_on)]# + datetime.timedelta(seconds=2.0)) ]
                ax.plot( post_laser_df['x'],
                         post_laser_df['y'],
                         'r.'
                         )
                pre_laser_df = t_df[ t_df.index <= (laser_on)]# + datetime.timedelta(seconds=2.0)) ]
                ax.plot( pre_laser_df['x'],
                         pre_laser_df['y'],
                         'k.'
                         )


    # how many trials do we want to analyze?
    condition = conditions[0]
    r =  dfs[conditions[0]]
    (_,parsed_data,_) = dfs[conditions[0]][0]
    if parsed_data['trialnum'] is None:
        trial_num_list = [1]
    else:
        trial_num_list = [1,2,3]

    # now iterate trial by trial
    for trial_num in trial_num_list:
        for measurement in ['proboscis',
                            'wing',
                            'jump',
                            'abdomen',
                            ]:

            counts = {}
            latency_values = defaultdict(list)

            n_rows, n_cols = my_subplot( n_conditions )
            if DO_LATENCY_FIGURE:
                fig = plt.figure('timeseries %s, trial %d'%(measurement,trial_num))
            df = {'latency':[],
                  'name_key':[],
                  }
            df_pooled = {'latency':[],
                         'name_key':[],
                         }
            for i, condition in enumerate(conditions):
                target_location, genotype = NAMES[condition]
                if DO_LATENCY_FIGURE:
                    ax = fig.add_subplot(n_rows, n_cols, i+1 )
                arrs = []
                total_trials = 0
                action_trials = 0
                mean_vels = []
                for (t_df,parsed_data,laser_on) in dfs[condition]:
                    if measurement not in t_df:
                        continue
                    if parsed_data['trialnum'] is not None:
                        if int(parsed_data['trialnum']) != trial_num:
                            continue
                    else:
                        if trial_num != 1:
                            continue

                    v_mm = np.sqrt(t_df['vx'].values**2 + t_df['vy'].values**2)
                    mean_v_mm = np.mean(v_mm)
                    mean_vels.append( mean_v_mm )

                    if DO_LATENCY_FIGURE:
                        ax.plot(t_df[measurement],
                                label='%s (fly %s, trial %s)'%(parsed_data['condition'],
                                                               parsed_data['condition_flynum'],
                                                               parsed_data['trialnum'],
                                                               ))
                    arrs.append( t_df['proboscis'] )
                    selected_df = t_df[ t_df[measurement] > 0.5 ]
                    if len(selected_df)==0:
                        this_latency = np.inf
                    else:
                        '''
                        # assume any behavior is caused by laser
                        first_behavior_time = selected_df.index[0]
                        this_latency = (first_behavior_time - laser_on).total_seconds()
                        '''
                        # assume temporal causality in latency measurement
                        behavior_times = selected_df.index
                        for this_bt in behavior_times:
                            this_latency = (this_bt - laser_on).total_seconds()
                            if this_latency >= 0:
                                break
                        if this_latency < 0:
                            this_latency = np.inf

                    latency_values[condition].append( this_latency )
                    df['latency'].append( this_latency )
                    df_pooled['latency'].append( this_latency )
                    name_key = '%s %s'%(genotype, target_location)
                    if '>' not in genotype:
                        pooled_name_key = 'controls '+target_location
                    else:
                        pooled_name_key = name_key
                    df['name_key'].append(name_key)
                    df_pooled['name_key'].append( pooled_name_key )

                    if np.any( t_df[measurement] > 0.5 ):
                        action_trials += 1
                    total_trials += 1
                counts[condition] = (action_trials,total_trials, np.mean(mean_vels))
                all_arrs = np.array( arrs )
                mean_arr = np.mean( all_arrs, axis=0 )
                if DO_LATENCY_FIGURE:
                    ax.plot( mean_arr, color='k', lw=2 )
                    ax.set_title('%s (n=%d)'%(condition, counts[condition][1]))
                    ax.legend()
            if DO_LATENCY_FIGURE:
                fig.subplots_adjust(hspace=0.39)

            df = pd.DataFrame(df)
            df_fname = '%s_notpooled.df'%(measurement,)
            df.to_pickle(df_fname)
            print 'saved',df_fname

            df_pooled = pd.DataFrame(df_pooled)
            df_fname = '%s_pooled.df'%(measurement,)
            df_pooled.to_pickle(df_fname)
            print 'saved',df_fname

        print '='*80

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to mp4s')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--bagdir', type=str, default=None)
    args = parser.parse_args()
    dirname = args.path[0]

    if args.bagdir is None:
        args.bagdir = os.path.join(dirname,'bags')

    arena = madplot.Arena('mm')

    if args.only_plot:
        cache_fname = make_cache_fname(arena, dirname, args.bagdir, smooth=args.smooth)
        print 'loading cache',cache_fname
        with open(cache_fname,mode='rb') as f:
            data = pickle.load(f)
    else:
        data = prepare_data(arena, dirname, args.bagdir, smooth=args.smooth)

    plot_data( arena, dirname, args.smooth, data)

    if args.show:
        plt.show()
