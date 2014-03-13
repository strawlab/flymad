import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import sys
import glob
import datetime
import pprint

import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset

import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import madplot
import fake_plotly
from scipy.stats import kruskal

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0" ,  "0.12.0")

EXPERIMENT_DURATION = 70.0

def prepare_data(path, arena, smoothstr, smooth, medfilt, gts):

    pooldf = DataFrame()
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        cache_args = os.path.basename(csvfile), arena, smoothstr
        cache_fname = csvfile+'.madplot-cache'

        results = madplot.load_bagfile_cache(cache_args, cache_fname)
        if results is None:
            results = flymad_analysis.load_and_smooth_csv(csvfile, arena, smooth)
            if results is not None:
                #update the cache
                madplot.save_bagfile_cache(results, cache_args, cache_fname)
            else:
                print "skipping", csvfile
                continue

        df,dt,experimentID,date,time,genotype,laser,repID = results

        #we plot head v thorax v nolaser (so for the same of plotting, consider
        #these the genotypes
        genotype = genotype + '-' + laser

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        if 0:
            fig = plt.figure()
            fig.suptitle(os.path.basename(csvfile))
            ax = fig.add_subplot(1,1,1)
            df['experiment'] = 1
            df['tobj_id'] = 1
            madplot.plot_tracked_trajectory(ax, df, arena,
                        debug_plot=False,
                        color='k',
            )
            ax.add_patch(arena.get_patch(color='k', alpha=0.1))

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < EXPERIMENT_DURATION:
            print "\tmissing data", csvfilefn
            continue

        print "\t%ss experiment" % duration

        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 300] = np.nan
        df['v'] = df['v'].fillna(method='ffill')

        #median filter
        if medfilt:
            df['v'] = scipy.signal.medfilt(df['v'].values, medfilt)

        #Here we have a 10ms resampled dataframe at least EXPERIMENT_DURATION seconds long.
        df = df.head(flymad_analysis.get_num_rows(EXPERIMENT_DURATION))
        tb = flymad_analysis.get_resampled_timebase(EXPERIMENT_DURATION)
        #find when the laser first came on (argmax returns the first true value if
        #all values are identical
        t0idx = np.argmax(np.gradient(df['laser_state'].values > 0))
        t0 = tb[t0idx]
        df['t'] = tb - t0

        #groupby on float times is slow. make a special align column 
        df['t_align'] = np.array(range(0,len(df))) - t0idx

        df['obj_id'] = flymad_analysis.create_object_id(date,time)
        df['Genotype'] = genotype
        df['lasergroup'] = laser

        pooldf = pd.concat([pooldf, df])

    data = {}
    for gt in gts:
        gtdf = pooldf[pooldf['Genotype'] == gt]

        lgs = gtdf['lasergroup'].unique()
        if len(lgs) != 1:
            raise Exception("only one lasergroup handled for gt %s: not %s" % (
                             gt, lgs))

        grouped = gtdf.groupby(['t_align'], as_index=False)
        data[gt] = dict(mean=grouped.mean().astype(float),
                        std=grouped.std().astype(float),
                        n=grouped.count().astype(float),
                        first=grouped.first(),
                        df=gtdf)

    return data

def get_stats(group):
    return {'mean': group.mean(),
            'var' : group.var(),
            'n' : group.count()
           }

def add_obj_id(df,align_colname='t_align'):
    results = np.zeros( (len(df),), dtype=np.int )
    obj_id = 0
    for i,(ix,row) in enumerate(df.iterrows()):
        if row[align_colname]==0.0:
            obj_id += 1
        results[i] = obj_id
    df['obj_id']=results
    return df

def calc_p_values(data, gt1_name, gt2_name, align_colname='t_align', stat_colname='v'):
    df_ctrl = data[gt1_name]['df']
    df_exp = data[gt2_name]['df']

    df_ctrl = add_obj_id(df_ctrl,align_colname=align_colname)
    df_exp = add_obj_id(df_exp,align_colname=align_colname)

    align_start = df_ctrl[align_colname].min()
    dalign = df_ctrl[align_colname].max() - align_start

    p_values = DataFrame()
    binsize = 50 # 1 sec

    bins = np.linspace(0,dalign,binsize) + align_start
    binned_ctrl = pd.cut(df_ctrl[align_colname], bins, labels= bins[:-1])
    binned_exp = pd.cut(df_exp[align_colname], bins, labels= bins[:-1])
    for x in binned_ctrl.levels:
        test1_all_flies_df = df_ctrl[binned_ctrl == x]
        bin_start_time = test1_all_flies_df['t'].min()
        bin_stop_time = test1_all_flies_df['t'].max()

        test1 = []
        for obj_id, fly_group in test1_all_flies_df.groupby('obj_id'):
            test1.append( np.mean(fly_group[stat_colname].values) )
        test1 = np.array(test1)

        test2_all_flies_df = df_exp[binned_exp == x]
        test2 = []
        for obj_id, fly_group in test2_all_flies_df.groupby('obj_id'):
            test2.append( np.mean(fly_group[stat_colname].values) )
        test2 = np.array(test2)

        if len(test1)<=5 or len(test2)<=5:
            # reaching end of data - stop
            break
        hval, pval = kruskal(test1, test2)
        dftemp = DataFrame({'Bin_number': x,
                            'P': pval,
                            'bin_start_time':bin_start_time,
                            'bin_stop_time':bin_stop_time,
                            'name1':gt1_name,
                            'name2':gt2_name,
                            'test1_n':len(test1),
                            'test2_n':len(test2),
                            }, index=[x])
        p_values = pd.concat([p_values, dftemp])
    return p_values

def get_pairwise(data,gt1_name,gt2_name):
    p_values = calc_p_values(data, gt1_name, gt2_name,
                             align_colname='t_align', stat_colname='v')

    starts = np.array(p_values['bin_start_time'].values)
    stops = np.array(p_values['bin_stop_time'].values)
    pvals = p_values['P'].values
    n1 = p_values['test1_n'].values
    n2 = p_values['test2_n'].values
    logs = -np.log10(pvals)

    xs = []
    ys = []
    texts = []

    for i in range(len(logs)):
        xs.append( starts[i] )
        ys.append( logs[i] )
        texts.append( 'p=%.3g, n=%d,%d t=%s to %s'%(
            pvals[i], n1[i], n2[i], starts[i], stops[i] ) )

        xs.append( stops[i] )
        ys.append( logs[i] )
        texts.append( '')

    this_dict = {
        'name':'%s vs. %s'%(p_values['name1'][0], p_values['name2'][0]),
        'x':[float(x) for x in xs],
        'y':[float(y) for y in ys],
        'text':texts,
        }

    layout = {
        'xaxis': {'title': 'Time (s)'},
        'yaxis': {'title': '-Log10(p)'},
        }
    results = {'data':this_dict,
               'layout':layout,
               }
    return results

def view_pairwise_stats_plotly( data, names ):
    pairs = []
    for i,name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if j<=i:
                continue
            pairs.append( (name1, name2 ) )

    graph_data = []
    for pair in pairs:
        name1, name2 = pair
        pairwise_data = get_pairwise( data, name1, name2 )
        graph_data.append( pairwise_data['data'] )

    if len( graph_data )==0:
        return

    # sign up and get from https://plot.ly
    plotly_username = os.environ.get('PLOTLY_USERNAME',None)
    plotly_api_key = os.environ.get('PLOTLY_API_KEY',None)

    if plotly_username is None:
        print 'environment variable PLOTLY_USERNAME not set. Will not use plotly'
    else:
        # plotly
        # developed with plotly 0.5.11
        import plotly
        py = plotly.plotly(plotly_username, plotly_api_key)

        result = py.plot( graph_data, layout=pairwise_data['layout'] )
        pprint.pprint( result )

    result2 = fake_plotly.plot( graph_data, layout=pairwise_data['layout'] )
    pprint.pprint(result2)
    fig_fname = 'p_values_csv_speed.png'
    fullpath = flymad_plot.get_plotpath(path,fig_fname)
    result2['fig'].savefig(fullpath)
    print 'saved',fig_fname

def plot_data(path, data, arena, note):

    LABELS = {'OK371shits-130h':'OK371>ShibireTS (head)',
              'OK371shits-nolaser':'Control',
              'OK371shits-130t':'OK371>ShibireTS (thorax)',
    }

    COLORS = {'OK371shits-130h':flymad_plot.RED,
              'OK371shits-nolaser':flymad_plot.BLACK,
              'OK371shits-130t':flymad_plot.ORANGE,
    }


    fig2 = plt.figure("Speed")
    ax = fig2.add_subplot(1,1,1)

    datasets = {}
    for gt in data:
        gtdf = data[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['v'].values,
                            std=gtdf['std']['v'].values,
                            n=gtdf['n']['v'].values,
                            label=LABELS[gt],
                            color=COLORS[gt],
                            df=gtdf['df'],
                            N=len(gtdf['df']['obj_id'].unique()))

    ctrlmean = data['OK371shits-nolaser']['mean']

    result_d = flymad_plot.plot_timeseries_with_activation(ax,
                targetbetween=dict(xaxis=ctrlmean['t'].values,
                                   where=ctrlmean['laser_state'].values>0),
                downsample=5,
                sem=True,
                note="OK371shits\n%s\n" % note,
                individual={k:{'groupby':'obj_id','xaxis':'t','yaxis':'v'} for k in datasets},
                individual_title='Speed Individual Traces',
                return_dict=True,
                **datasets
    )
    figs = result_d['figs']

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (%s/s)' % arena.unit)
    ax.set_xlim([-20, 30])
    ax.set_ylim([0, 18])

    flymad_plot.retick_relabel_axis(ax, [-20, 0, 10, 20], [0, 5, 10, 15])

    fig2.savefig(flymad_plot.get_plotpath(path,"speed_plot.png"), bbox_inches='tight')
    fig2.savefig(flymad_plot.get_plotpath(path,"speed_plot.svg"), bbox_inches='tight')

    for efigname, efig in figs.iteritems():
        efig.savefig(flymad_plot.get_plotpath(path,"speed_plot_individual_%s.png" % efigname), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shits-130h'
    CTRL_GENOTYPE = 'OK371shits-nolaser'
    EXP2_GENOTYPE = 'OK371shits-130t'

    CALIBRATION_FILE = 'calibration20140219_064948.filtered.yaml'
    GENOTYPES = [EXP_GENOTYPE, CTRL_GENOTYPE, EXP2_GENOTYPE]

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth', default=True)
    parser.add_argument('--calibration-dir', help='calibration directory containing yaml files', required=True)

    args = parser.parse_args()
    path = args.path[0]

    medfilt = 51
    smoothstr = '%s' % {True:'smooth',False:'nosmooth'}[args.smooth]

    calibration_file = os.path.join(args.calibration_dir, CALIBRATION_FILE)
    arena = madplot.Arena(
                'mm',
                **flymad_analysis.get_arena_conf(calibration_file=calibration_file))

    note = "%s %s\n%r\nmedfilt %s" % (arena.unit, smoothstr, arena, medfilt)

    cache_fname = os.path.join(path,'speed.madplot-cache')
    cache_args = (path, arena, smoothstr, args.smooth, medfilt, GENOTYPES)
    data = None
    if args.only_plot:
        data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if data is None:
        data = prepare_data(path, arena, smoothstr, args.smooth, medfilt, GENOTYPES)
        madplot.save_bagfile_cache(data, cache_args, cache_fname)

    view_pairwise_stats_plotly(data, [EXP_GENOTYPE,
                                      CTRL_GENOTYPE,
                                      EXP2_GENOTYPE] )
    plot_data(path, data, arena, note)

    if args.show:
        plt.show()

