#!/usr/bin/env python
import re
import warnings
import os.path
import collections
import glob
import pprint, copy
import datetime
import math
import cPickle as pickle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.transforms as mtransforms
import strawlab_mpl.defaults as smd
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

from pairs2groups import label_homogeneous_groups, label_homogeneous_groups_pandas # github.com/astraw/pairs2groups

import roslib; roslib.load_manifest('flymad')
import rosbag

import scipy.signal
import scipy.stats
import scipy.interpolate

R2D = 180/np.pi
CACHE_FNAME = 'optodata.pkl'

def setup_defaults():
    rcParams = matplotlib.rcParams

    rcParams['legend.numpoints'] = 1
    rcParams['legend.fontsize'] = 'medium' #same as axis
    rcParams['legend.frameon'] = False
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1
    matplotlib.rc('font', size=8)

def wrap_dtheta_plus_minus(dtheta, around=np.pi):
    """
    wraps an angle over the interval [-around,around)

    e.g. wrap_dtheta_plus_minus(dt, np.pi) returns an angle
    wrapped [-pi,pi)
    """

    if (dtheta>0):
        return math.fmod(dtheta+around, 2.0*around) - around
    else:
        return math.fmod(dtheta-around, 2.0*around) + around

def calc_dtheta(df):
    #naieve method for calculating dtheta assuming wraparound.

    dtheta = []
    dtheta_idx = []

    for i0,i1 in madplot.pairwise(df.iterrows()):
        t0idx,t0row = i0
        t1idx,t1row = i1
        t0 = t0row['theta']
        t1 = t1row['theta']

        dt = (t1-t0)
        #dtheta ranges from -pi/2.0 -> pi/2.0
        dt = wrap_dtheta_plus_minus(dt, around=np.pi/2.0)

        dtheta.append(dt)
        dtheta_idx.append(t0idx)

    return pd.Series(dtheta,index=dtheta_idx)

def choose_orientations_primitive(theta):
    """
    Remove head/tail ambiguity.

    Assumes theta is == orientation modulo pi, in other words,
    that the orientation is ambiguous with respect to head/tail.
    """
    theta_prev = theta[0]
    result = [theta_prev]
    diffs = np.array([-3*np.pi,-2*np.pi,-np.pi,0,np.pi,2*np.pi,3*np.pi])
    PI2 = 2*np.pi
    for i, ambiguous_angle in enumerate( theta[1:] ):
        possible_angles = diffs + ambiguous_angle
        rotation_distance = abs( theta_prev - possible_angles)
        diff_idx = np.argmin(rotation_distance)

        # update for next
        theta_prev = ambiguous_angle+diffs[diff_idx]
        if theta_prev < 0:
            theta_prev = theta_prev + PI2
        elif theta_prev >= PI2:
            theta_prev = theta_prev - PI2
        result.append( theta_prev )
    result = np.array(result)
    assert len(result)==len(theta)
    return result

def supplement_times(df):
    """create new column ('time_since_start') in DataFrame df"""
    times = df['t_ts'].values
    good_cond = ~np.isnan(times)
    good_times = times[good_cond]
    t0 = times[good_cond][0]
    df['time_since_start'] = df['t_ts']-t0

def supplement_angles(df,
                      cutoff_hz=5.0,
                      filter_order=8,
                      method='downsampled',
                      downsample_sec=1.0,
                      ):
    """create new column ('angular_velocity') in DataFrame df

    if cutoff_hz==0, no lowpass filtering is done.

    Assumes df['theta'] is == orientation modulo pi, in other words,
    that the orientation is ambiguous with respect to head/tail.

    Methods:
         'raw' - central difference on raw theta values
         'lowpassed' - central difference on lowpassed theta values
         'downsampled' - central difference on downsampled theta values
    """
    stamps = df['t_ts'].values
    bad_cond1 = np.isnan(stamps)
    bad_cond2 = np.isnan(df['theta'].values)

    bad_cond = bad_cond1 | bad_cond2

    good_cond = ~bad_cond
    goodstamps = stamps[good_cond]
    dgoodstamps = goodstamps[1:]-goodstamps[:-1]
    if 1:
        # check assumptions about pandas
        bad_cond2 = np.isnan(df['theta'].values)
        assert bad_cond.ndim == bad_cond2.ndim
        assert bad_cond.shape == bad_cond2.shape
        assert np.allclose( bad_cond, bad_cond2 )

        assert np.alltrue(dgoodstamps > 0)

    # unwrap theta, dealing with nans
    good_theta = df['theta'].values[good_cond]
    good_theta_chosen = choose_orientations_primitive(good_theta)
    good_theta_unwrapped = np.unwrap(good_theta_chosen)

    dt = np.mean(dgoodstamps)

    if cutoff_hz != 0:
        sample_rate_hz = 1.0/dt
        nyquist_rate_hz = sample_rate_hz*0.5

        filt_b, filt_a = scipy.signal.butter(filter_order,
                                             cutoff_hz/nyquist_rate_hz)
        lowpassed = scipy.signal.filtfilt(filt_b, filt_a,
                                          good_theta_unwrapped)

    # calculate angular velocity using central difference
    if method=='raw':
        delta_theta = np.gradient(good_theta_unwrapped)
        angular_velocity = delta_theta/dt
    elif method=='lowpassed':
        delta_theta = np.gradient(lowpassed)
        angular_velocity = delta_theta/dt
    elif method=='downsampled':
        # downsample theta
        n_samps = int(np.round(downsample_sec/dt))
        downdt = n_samps*dt
        downtime = goodstamps[::n_samps]
        downtheta = lowpassed[::n_samps]
        if 1:
            # linear interpolation of downsampled theta
            interp = scipy.interpolate.interp1d( downtime, downtheta,
                                                 bounds_error=False,
                                                 fill_value=np.nan,
                                                 )
            linear_theta_values = interp( goodstamps )
            delta_theta = np.gradient(linear_theta_values)

        downvel = np.gradient( downtheta )/downdt
        interp = scipy.interpolate.interp1d( downtime, downvel,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             )
        angular_velocity = interp( goodstamps )

    # now fill dataframe with computed values
    tmp = np.nan*np.ones( (len(df,) ))
    tmp[good_cond] = good_theta_chosen
    df['theta_wrapped'] = tmp
    tmp = np.nan*np.ones( (len(df,) ))
    tmp[good_cond] = good_theta_unwrapped
    df['theta_unwrapped'] = tmp
    tmp = np.nan*np.ones( (len(df,) ))
    tmp[good_cond] = lowpassed
    df['theta_unwrapped_lowpass'] = tmp
    if method=='downsampled':
        tmp = np.nan*np.ones( (len(df,) ))
        tmp[good_cond] = linear_theta_values
        df['theta_unwrapped_downsampled'] = tmp
    tmp = np.nan*np.ones( (len(df,) ))
    tmp[good_cond] = angular_velocity
    df['angular_velocity'] = tmp


def prepare_data(arena, path, smoothstr, smooth):
    # keep here to allow use('Agg') to work
    import matplotlib.pyplot as plt
    import madplot

    path = os.path.abspath(path)
    print 'opening bag files in %r'%path

    GENOTYPES = {"NINE":"*.bag",
                 "CSSHITS":"ctrl/CSSHITS/*.bag",
                 "NINGal4":"ctrl/NINGal4/*.bag"}

    CHUNKS = collections.OrderedDict()
    CHUNKS["1_loff+ve"] = (0,30)
    CHUNKS["2_loff-ve"] = (30,60)
    CHUNKS["3_lon-ve"] = (60,120)
    CHUNKS["4_lon+ve"] = (120,180)
    CHUNKS["5_loff+ve"] = (180,240)
    CHUNKS["6_loff-ve"] = (240,300)
    CHUNKS["7_loff+ve"] = (300,360)

    save_times = None

    data = {gt:dict(chunk={}) for gt in GENOTYPES}
    for gti,gt in enumerate(GENOTYPES):
        #if gti>=1: break
        pattern = os.path.join(path, GENOTYPES[gt])
        bags = glob.glob(pattern)

        for bag in bags:

            df = madplot.load_bagfile_single_dataframe(
                        bag, arena,
                        ffill=['laser_power','e_rotator_velocity_data'],
                        extra_topics={'/rotator/velocity':['data']},
                        smooth=smooth
            )
            supplement_angles(df)
            supplement_times(df)

            if 1:
                # John's category stuff

                #there might be multiple object_ids in the same bagfile for the same
                #fly, so calculate dtheta for all of them
                dts = []
                for obj_id, group in df.groupby('tobj_id'):
                    if np.isnan(obj_id):
                        continue
                    dts.append( calc_dtheta(group) )

                #join the dthetas vertically (axis=0) and insert as a column into
                #the full dataframe
                df['dtheta'] = pd.concat(dts, axis=0)


            good = ~np.isnan(df['time_since_start'])
            if save_times is None:
                save_times = np.arange(0,
                                       (df['time_since_start'][good].values[-1]+1e-6),
                                       1.0 ) # every second

            if 'timeseries_angular_vel' not in data[gt]:
                data[gt]['bag_fname'] = []
                data[gt]['timeseries_angular_vel'] = []
                data[gt]['timeseries_vel'] = []
                data[gt]['save_times'] = save_times
                good_stim = ~np.isnan(df['e_rotator_velocity_data'].values)
                good_both = good & good_stim
                data[gt]['stimulus_velocity'] = (
                    df['time_since_start'].values[good_both],
                    df['e_rotator_velocity_data'].values[good_both])
            if gt=='CSSHITS' and 'laser_power' not in data[gt]:
                interp = scipy.interpolate.interp1d( df['time_since_start'][good],
                                                     df['laser_power'][good],
                                                     bounds_error=False,
                                                     fill_value=np.nan,
                                                     )
                save_laser = interp(save_times)
                data[gt]['laser_power'] = save_laser

            interp = scipy.interpolate.interp1d( df['time_since_start'][good],
                                                 df['angular_velocity'][good],
                                                 bounds_error=False,
                                                 fill_value=np.nan,
                                                 )
            save_angular_vel = interp(save_times)
            data[gt]['timeseries_angular_vel'].append( save_angular_vel )
            data[gt]['bag_fname'].append( bag )

            interp = scipy.interpolate.interp1d( df['time_since_start'][good],
                                                 df['v'][good],
                                                 bounds_error=False,
                                                 fill_value=np.nan,
                                                 )
            save_vel = interp(save_times)
            data[gt]['timeseries_vel'].append( save_vel )

            # if 1:
            #     break

            #calculate mean_dtheta over each phase of the experiment
            t00 = df.index[0]
            for c in CHUNKS:
                s0,s1 = CHUNKS[c]

                t0 = datetime.timedelta(seconds=s0)
                t1 = datetime.timedelta(seconds=s1)

                mean_dtheta = df[t00+t0:t00+t1]["dtheta"].mean()
                mean_v = df[t00+t0:t00+t1]["v"].mean()

                if c not in data[gt]:
                    data[gt]['chunk'][c] = []
                data[gt]['chunk'][c].append( (mean_dtheta, mean_v) )

    pickle.dump(data, open(CACHE_FNAME,'wb'), -1)
    print 'saved cache to %s'%CACHE_FNAME
    return data

def plot_data(arena, path, smoothstr, data):
    import matplotlib.pyplot as plt

    COLORS = {"NINE":"r",
              "CSSHITS":"g",
              "NINGal4":"b",
              'pooled controls':'k'
              }

    # ---- pool controls ------------
    data['pooled controls'] = copy.deepcopy(data['CSSHITS'])
    assert np.allclose(data['NINGal4']['save_times'], data['pooled controls']['save_times'])


    # copy 'NINGal4' data into pooled controls
    for dname in data['NINGal4']:
        for row in data['NINGal4'][dname]:
            if dname in ['save_times','stimulus_velocity']:
                continue
            elif dname=='chunk':
                warnings.warn('not updating old chunk data for pooled controls')
            else:
                data['pooled controls'][dname].append(row)
    # for row in data['NINGal4']['chunk']:
    #     for key in row:
    #         data['pooled controls']['chunk'][key].append(row[key])

    # ----- raw timeseries plots ------------

    fig_angular_vel = plt.figure(figsize=(3.5, 2.0))
    fig_linear_vel = plt.figure(figsize=(3.5, 2.0))

    fig_summary_angular = plt.figure()
    ax_summary_angular = fig_summary_angular.add_subplot(211)
    ax_summary_linear = fig_summary_angular.add_subplot(212)
    pool_stats = collections.defaultdict(list)

    #order = ['NINE', 'CSSHITS', 'NINGal4']
    order = ['NINE', 'pooled controls']
    n_subplots = len(order)
    ax = None
    ax_angular_vel = fig_angular_vel.add_subplot(111)
    ax_angular_vel.axhline(0,color='black')
    ax_linear_vel = fig_linear_vel.add_subplot(111)

    did_stimulus_plot = False
    plot_angular_datasets = collections.OrderedDict()
    plot_linear_datasets = collections.OrderedDict()

    for gti,gt in enumerate(order):

        # --- calculate stimulus -----------
        stim_times, stim_vel = data[gt]['stimulus_velocity']
        if 1:
            '''
            # 10 revolutions take 25 seconds at standard speed
            real_stim_vel = 10*2*np.pi/25.0 # 10 revolutions/second
            print 'real_stim_vel',real_stim_vel
            print 'gain[0]',real_stim_vel/stim_vel[100]
            print 'gain[1]',real_stim_vel/stim_vel[1001]
            '''
            gain = 0.00083775 # found using above
            stim_vel = gain*stim_vel

        if not did_stimulus_plot:
            ax_angular_vel.plot( stim_times,
                                 stim_vel*R2D,'k',
                                 lw=0.5, label='stimulus')
            did_stimulus_plot = True

        transition_idxs = np.nonzero(abs(stim_vel[1:]-stim_vel[:-1]))[0]
        #print 'stim_times[transition_idxs]',stim_times[transition_idxs]
        n_transitions = len(transition_idxs)

        # ---- done calculating stimulus ---

        all_angular_vel_timeseries = np.array(data[gt]['timeseries_angular_vel'])
        all_linear_vel_timeseries = np.array(data[gt]['timeseries_vel'])
        times = data[gt]['save_times']
        mean_angular_timeseries = np.mean( all_angular_vel_timeseries, axis=0 )
        error_angular_timeseries = np.std( all_angular_vel_timeseries, axis=0 )
        mean_linear_timeseries = np.mean( all_linear_vel_timeseries, axis=0 )
        error_linear_timeseries = np.std( all_linear_vel_timeseries, axis=0 )

        if 1:
            bin_data = []
            ang_means = []
            ang_stds = []
            lin_means = []
            lin_stds = []
            for i in range(n_transitions-1):
                this_data = dict(index=i,
                                 angular_velocity_values=[],
                                 linear_velocity_values=[],
                                 )
                start_time = stim_times[transition_idxs[i]]
                stop_time = stim_times[transition_idxs[i+1]]
                print 'start_time, stop_time',start_time, stop_time
                start_idx = np.argmin(abs(times - start_time))
                stop_idx =  np.argmin(abs(times - stop_time))
                print 'times[start_idx], times[stop_idx]',times[start_idx], times[stop_idx]
                this_data['start_time'] = times[start_idx]
                this_data['stop_time'] = times[stop_idx]
                assert len( all_angular_vel_timeseries)== len(all_linear_vel_timeseries)
                for j in range( len(all_angular_vel_timeseries)):
                    this_angular_timeseries = all_angular_vel_timeseries[j]
                    this_linear_timeseries =  all_linear_vel_timeseries[j]

                    ang = np.mean(this_angular_timeseries[start_idx:stop_idx])
                    lin = np.mean(this_linear_timeseries[start_idx:stop_idx])

                    this_data['angular_velocity_values'].append( ang )
                    this_data['linear_velocity_values'].append( lin )

                    pool_stats['gt'].append(gt)
                    pool_stats['bin_number'].append(i)
                    pool_stats['bin_start'].append( this_data['start_time'] )
                    pool_stats['bin_stop'].append( this_data['stop_time'] )
                    pool_stats['nth_fly_of_gt'].append(j)
                    pool_stats['mean_angular_velocity_during_bin'].append( ang )
                    pool_stats['mean_linear_velocity_during_bin'].append( lin )

                this_data[ 'ang_mean' ] = np.mean( this_data['angular_velocity_values'] )
                this_data[ 'ang_std' ] = np.std( this_data['angular_velocity_values'] )
                this_data[ 'lin_mean' ] = np.mean( this_data['linear_velocity_values'] )
                this_data[ 'lin_std' ] = np.std( this_data['linear_velocity_values'] )
                bin_data.append( this_data )
                ang_means.append( this_data['ang_mean'] )
                ang_stds.append( this_data['ang_std'] )
                lin_means.append( this_data['lin_mean'] )
                lin_stds.append( this_data['lin_std'] )

            ang_means = np.array(ang_means)
            ang_stds  = np.array(ang_stds)
            lin_means = np.array(lin_means)
            lin_stds  = np.array(lin_stds)

            ax_summary_angular.axhline(0,color='black')
            ax_summary_angular.plot( ang_means*R2D, label=gt, color=COLORS[gt] )
            ax_summary_angular.plot( (ang_means+ang_stds)*R2D, color=COLORS[gt] )
            ax_summary_angular.plot( (ang_means-ang_stds)*R2D, color=COLORS[gt] )

            ax_summary_linear.axhline(0,color='black')
            ax_summary_linear.plot( lin_means, label=gt, color=COLORS[gt] )
            ax_summary_linear.plot( lin_means+lin_stds, color=COLORS[gt] )
            ax_summary_linear.plot( lin_means-lin_stds, color=COLORS[gt] )

        this_data_angular_vel = {'xaxis':times,
                                 'value':mean_angular_timeseries*R2D,
                                 'std':error_angular_timeseries*R2D,
                                 'label':gt,
                                 'color':COLORS[gt],
                                 }
        this_data_linear_vel = {'xaxis':times,
                                'value':mean_linear_timeseries,
                                'std':error_linear_timeseries,
                                'label':gt,
                                'color':COLORS[gt],
                                }
        plot_angular_datasets[gt]=this_data_angular_vel
        plot_linear_datasets[gt]=this_data_linear_vel

    from flymad.flymad_plot import plot_timeseries_with_activation
    tb = {'xaxis':data['pooled controls']['save_times'],
          'where':data['pooled controls']['laser_power']>1,
          }
    plot_timeseries_with_activation( ax_angular_vel,
                                     downsample=2,
                                     targetbetween=[tb],
                                     **plot_angular_datasets)
    ax_angular_vel.set_xticks([0,180,360])
    ax_angular_vel.set_yticks([-200,0,200])
    ax_angular_vel.set_ylim(-200,200)
    ax_angular_vel.spines['left'].set_bounds(-200,200.0)
    ax_angular_vel.spines['bottom'].set_bounds(0,360.0)
    ax_angular_vel.spines['bottom'].set_linewidth(0.3)
    ax_angular_vel.spines['left'].set_linewidth(0.3)
    ax_angular_vel.set_xlabel('Time (s)')
    ax_angular_vel.set_ylabel('Angular velocity (deg/s)')
    fig_angular_vel.subplots_adjust(left=0.2,bottom=0.23) # do not clip text
    fig_fname = 'fig_ts_angular_vel.png'
    fig_angular_vel.savefig(fig_fname)
    print 'saved',fig_fname
    fig_fname = 'fig_ts_angular_vel.svg'
    fig_angular_vel.savefig(fig_fname)
    print 'saved',fig_fname
    fig_fname = 'fig_ts_angular_vel.pdf'
    fig_angular_vel.savefig(fig_fname)
    print 'saved',fig_fname



    plot_timeseries_with_activation( ax_linear_vel,
                                     targetbetween=[tb],
                                     downsample=3,
                                     **plot_linear_datasets)
    ax_linear_vel.set_xticks([0,180,360])
    ax_linear_vel.set_yticks([0,20,40])
    ax_linear_vel.set_ylim([0,40])
    ax_linear_vel.spines['bottom'].set_bounds(0,360.0)
    ax_linear_vel.spines['bottom'].set_linewidth(0.3)
    ax_linear_vel.spines['left'].set_linewidth(0.3)
    ax_linear_vel.set_xlabel('Time (s)')
    ax_linear_vel.set_ylabel('Velocity (mm/s)')
    fig_linear_vel.subplots_adjust(left=0.2,bottom=0.23) # do not clip text
    fig_fname = 'fig_ts_linear_vel.png'
    fig_linear_vel.savefig(fig_fname)
    print 'saved',fig_fname
    fig_fname = 'fig_ts_linear_vel.svg'
    fig_linear_vel.savefig(fig_fname)
    print 'saved',fig_fname
    fig_fname = 'fig_ts_linear_vel.pdf'
    fig_linear_vel.savefig(fig_fname)
    print 'saved',fig_fname

    if 1:

        def p2star(p):
            if   p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''
        # calculate p values
        pool_stats_df = pd.DataFrame(pool_stats)
        df_fname = 'optomotor_stats.df'
        pool_stats_df.to_pickle(df_fname)
        sig = do_stats( df_fname )
        for i in range(len(sig['bin_num'])):
            bin_num = sig['bin_num'][i]
            p_value = sig['ang'][i]
            ax_summary_angular.text( bin_num,
                                     0,
                                     p2star(p_value))
            p_value = sig['ang_zero'][i]
            ax_summary_angular.text( bin_num,
                                     -100,
                                     p2star(p_value))

            p_value = sig['lin'][i]
            ax_summary_linear.text( bin_num,
                                    0,
                                    p2star(p_value))

    fig_fname = 'fig_summary.svg'
    fig_summary_angular.savefig(fig_fname)
    print 'saved',fig_fname
    fig_fname = 'fig_summary.pdf'
    fig_summary_angular.savefig(fig_fname)
    print 'saved',fig_fname


    # -----------------------

    # plot individual timeseries

    for gti,gt in enumerate(order):
        fig_gt = plt.figure('indiv_timeseries_'+gt)
        n_rows = len(data[gt]['timeseries_vel'])
        n_cols = 2 # linear and angular
        save_times = data[gt]['save_times']
        ax_lin = None
        ax_ang = None
        for i in range(n_rows):
            angular_vel = data[gt]['timeseries_angular_vel'][i]
            linear_vel = data[gt]['timeseries_vel'][i]

            ax_lin = fig_gt.add_subplot( n_rows, 2, i*2+1, sharex=ax_lin, sharey=ax_lin )
            ax_ang = fig_gt.add_subplot( n_rows, 2, i*2+2, sharex=ax_ang, sharey=ax_ang )
            ax_ang.axhline(0,color='black')

            ax_lin.plot( save_times, linear_vel )
            ax_ang.plot( save_times, angular_vel*R2D )

            bag = os.path.basename(data[gt]['bag_fname'][i])
            ax_lin.text(50,50,'lin '+bag)
            ax_ang.text(50,3*R2D,'ang '+bag)

            ax_lin.set_ylim(0,50)
            ax_ang.set_ylim(-300,300)
        plt.figtext(0.05, 0.05, gt)
        fname = 'individual_%s.pdf'%gt
        fig_gt.savefig(fname)
        print 'saved',fname

    # -----------------------

    if 1:
        return

    # - John's category stuff ----------------

    #pprint.pprint(data)

    figt = plt.figure('dtheta %s' % (smoothstr), figsize=(16,10))
    axt = figt.add_subplot(1,1,1)
    figv = plt.figure('v %s' % (smoothstr), figsize=(16,10))
    axv = figv.add_subplot(1,1,1)


    for gt in order:
        xdata = []
        ytdata = []
        yvdata = []
        for xlabel in sorted(data[gt]['chunk']):
            xloc = int(xlabel[0])
            for mean_dtheta,mean_v in data[gt]['chunk'][xlabel]:
                xdata.append(xloc)
                ytdata.append(mean_dtheta)
                yvdata.append(mean_v)

#        print gt,xdata,ydata

        axt.plot(xdata,ytdata,'o',color=COLORS[gt],markersize=5,label=gt)
        axv.plot(xdata,yvdata,'o',color=COLORS[gt],markersize=5,label=gt)

    for ax in (axt,axv):
        ax.legend()
        ax.set_xlim(0,9)

        ax.set_xticks([int(s[0]) for s in sorted(data["NINE"]['chunk'])])
        ax.set_xticklabels([s[2:] for s in sorted(data["NINE"]['chunk'])])

    figpath = os.path.join(path,'dtheta_%s.png' % (smoothstr))
    figt.savefig(figpath)
    print "wrote", figpath
    figpath = os.path.join(path,'v_%s.png' % (smoothstr))
    figv.savefig(figpath)
    print "wrote", figpath


def do_stats( df_fname ):
    df = pd.read_pickle(df_fname)

    assert len(df['gt'].unique())==2

    sig = dict(lin=[], # pvalues for difference between genotypes
               ang=[], # pvalues for difference between genotypes
               ang_zero=[], # pvalues for NINE different than zero
               bin_num=[],
               )

    for bin_number, group1 in df.groupby("bin_number",sort=True):
        sig['bin_num'].append(bin_number)
        print 'bin %d data ------------'%(bin_number,)
        print '  angular'
        print group1
        group_info = label_homogeneous_groups_pandas( group1,
                                                      groupby_column_name='gt',
                                                      value_column_name='mean_angular_velocity_during_bin')
        pprint.pprint(group_info)
        sig['ang'].append( group_info['p_values'][0,1] )

        if 1:
            # are the values for the experimental genotype different than zero?
            df_exp = group1[ group1['gt']=='NINE' ]
            v = df_exp['mean_angular_velocity_during_bin'].values
            v.shape = len(v),1 # reshape
            t,p = scipy.stats.ttest_1samp(v, # test if different than zero
                                          0.0)
            print 'p for %s: %.3f'%(group1['mean_angular_velocity_during_bin'].values,p)
            sig['ang_zero'].append(p)

        print '  linear'
        group_info = label_homogeneous_groups_pandas( group1,
                                                      groupby_column_name='gt',
                                                      value_column_name='mean_linear_velocity_during_bin')
        pprint.pprint(group_info)
        sig['lin'].append( group_info['p_values'][0,1] )
    return sig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to bag files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)

    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    smd.setup_defaults()
    setup_defaults()

    path = os.path.abspath(args.path[0])

    print 'bagfiles in',path
    assert os.path.isdir(path)

    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    import madplot # keep here to allow use('Agg') to work
    arena = madplot.Arena('mm')

    if args.only_plot:
        print 'loading cache',CACHE_FNAME
        with open(CACHE_FNAME,mode='rb') as f:
            data = pickle.load(f)
    else:
        data = prepare_data(arena, path, smoothstr, args.smooth)

    plot_data(arena, path, smoothstr, data)

    if args.show:
        import matplotlib.pyplot as plt # keep here to allow use('Agg') to work
        plt.show()
