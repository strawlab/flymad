import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import glob
import pickle
import re
import collections
import lifelines # http://lifelines.readthedocs.org, developed with v 0.2.3.0.7
from lifelines.statistics import logrank_test, pairwise_logrank_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg

from scipy.stats import ttest_ind
from scipy.stats import kruskal

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import madplot
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0", "0.12.0")

HEAD    = -100
THORAX  = +100
OFF     = 0

COLORS = {HEAD:flymad_plot.RED,
          THORAX:flymad_plot.ORANGE,
          }

EXPERIMENT_DURATION = 200.0

def p2stars(pval):
    result = ''
    if pval < 0.05:
        result = '*'
    if pval < 0.01:
        result = '**'
    if pval < 0.001:
        result = '***'
    return result

def prepare_data(path, resample_bin, gts):

    LASER_THORAX_MAP = {True:THORAX,False:HEAD}

    #PROCESS SCORE FILES:
    pooldf = pd.DataFrame()
    for df,metadata in flymad_analysis.load_courtship_csv(path):
        csvfilefn,experimentID,date,time,genotype,laser,repID = metadata

        dlaser = np.gradient(df['laser_state'].values)
        num_on_periods = (dlaser == 0.5).sum()
        if num_on_periods != 12:
            print "\tskipping file %s (%d laser on periods)" % (csvfilefn, num_on_periods/2)
            continue

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration < EXPERIMENT_DURATION:
            print "\tmissing data", csvfilefn
            continue
        print "\t%ss experiment" % duration

        #make new columns that indicates HEAD/THORAX targeting
        thorax = True
        laser_state = False

        trg = []
        for i0,i1 in madplot.pairwise(df.iterrows()):
            t0idx,t0row = i0
            t1idx,t1row = i1
            if t1row['laser_state'] >= 0.5 and t0row['laser_state'] == 0:
                thorax ^= True
                laser_state = True
            elif t0row['laser_state'] >= 0.5 and t1row['laser_state'] == 0:
                laser_state = False
            trg.append(OFF if not laser_state else LASER_THORAX_MAP[thorax])
        trg.append(OFF)
        df['ttm'] = trg

        #resample into 5S bins
        df = df.resample(resample_bin, fill_method='ffill')
        #trim dataframe
        df = df.head(flymad_analysis.get_num_rows(EXPERIMENT_DURATION, resample_bin))
        tb = flymad_analysis.get_resampled_timebase(EXPERIMENT_DURATION, resample_bin)

        #fix cols due to resampling
        df['laser_state'][df['laser_state'] > 0] = 1
        df['zx_binary'] = (df['zx'] > 0).values.astype(float)
        df['ttm'][df['ttm'] < 0] = HEAD
        df['ttm'][df['ttm'] > 0] = THORAX

        dlaser = np.gradient( (df['laser_state'].values > 0).astype(int) ) > 0
        t0idx = np.argmax(dlaser)
        t0 = tb[t0idx-1]
        df['t'] = tb - t0

        #groupby on float times is slow. make a special align column
        df['t_align'] = np.array(range(0,len(df))) - t0idx

        df['obj_id'] = flymad_analysis.create_object_id(date,time)
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        pooldf = pd.concat([pooldf, df])

    data = {}
    for gt in gts:
        gtdf = pooldf[pooldf['Genotype'] == gt]

        lgs = gtdf['lasergroup'].unique()
        if len(lgs) != 1:
            raise Exception("only one lasergroup handled for gt %s: not %s" % (
                             gt, lgs))

        grouped = gtdf.groupby(['t'], as_index=False)
        data[gt] = dict(mean=grouped.mean().astype(float),
                        std=grouped.std().astype(float),
                        n=grouped.count().astype(float),
                        first=grouped.first(),
                        df=gtdf)

    return data

def _update_latencies(df, sdx_e, sdx_g, sdx_c, name):
    # copy inputs
    sdx_e = sdx_e[:]
    sdx_g = sdx_g[:]
    sdx_c = sdx_c[:]

    VALID = True
    INVALID = False

    for obj_id, odf in df.groupby(['obj_id']):
        t0 = None
        found = False
        for tval, tdf in odf.groupby(['t']):
            if t0 is None:
                t0 = tval
            pulse_t = tval-t0 # time within current pulse
            if tdf['zx']>0:
                found = True
                sdx_e.append( pulse_t )
                sdx_g.append( name )
                sdx_c.append( VALID )
                break
        if not found:
            sdx_e.append( 1000.0 )
            sdx_g.append( name )
            sdx_c.append( INVALID )
    return sdx_e, sdx_g, sdx_c

def _do_group(df):
    t = []
    pulse_t = []
    all_cum = []
    all_n = []
    frac = []
    t0 = None
    for tval, tdf in df.groupby(['t'], as_index=False):
        t.append( tval )
        if t0 is None:
            t0 = tval
        pulse_t.append( tval-t0 )

        cum_ind = tdf['cum_wei_this_trial'].values
        cum = np.sum(cum_ind)
        n = len(cum_ind)
        all_cum.append( cum )
        all_n.append( n)
        frac.append( float(cum)/n )
    return dict(frac=np.array(frac),
                pulse_t=np.array(pulse_t),
                cum=np.array(all_cum),
                n=np.array(all_n),
                t=np.array(t))

def plot_cum( ax, data, **kw):
    t = data['pulse_t']
    frac = data['frac']

    xs = [0]
    ys = [0]
    for i in range(len(t)-1):
        xs.append( t[i] )
        ys.append( frac[i] )

        xs.append( t[i+1] )
        ys.append( frac[i] )
    ax.plot( xs, 100.0*np.array(ys), **kw )

def combine_cum_data(list_of_dicts):
    cum_by_pulse_t = collections.defaultdict(list)
    n_by_pulse_t = collections.defaultdict(list)
    for this_dict in list_of_dicts:
        pulse_t = this_dict['pulse_t']
        cum = this_dict['cum']
        n = this_dict['n']
        for i in range(len(pulse_t)):
            pulse_t[i]
            cum[i]
            n[i]
            cum_by_pulse_t[pulse_t[i]].append( cum[i] )
            n_by_pulse_t[pulse_t[i]].append( n[i] )
    pulse_ts = cum_by_pulse_t.keys()
    pulse_ts.sort()
    cum = []
    n = []
    frac = []
    for i in range(len(pulse_ts)):
        cum.append( np.sum(cum_by_pulse_t[ pulse_ts[i] ] ))
        n.append( np.sum(n_by_pulse_t[ pulse_ts[i] ] ))
        frac.append( float(cum[-1]) / n[-1] )
    return dict(frac=np.array(frac),
                pulse_t=np.array(pulse_t),
                cum=np.array(cum),
                n=np.array(n))

def do_cum_incidence(gtdf,label):
    # Calculate cumulative wing extension since last laser on.
    # (This is ugly...)

    gtdf['cum_wei_this_trial'] = 0
    gtdf['head_trial'] = 0
    gtdf['thorax_trial'] = 0
    obj_ids = gtdf['obj_id'].unique()
    for obj_id in obj_ids:
        prev_laser_state = 0
        cur_cum_this_trial = 0
        cur_head_trial = 0
        cur_thorax_trial = 0
        cur_ttm = 0
        cur_state = 'none'
        prev_t = -np.inf
        for rowi in range(len(gtdf)):
            row = gtdf.iloc[rowi]
            if row['obj_id'] != obj_id:
                continue
            assert row['t'] > prev_t
            prev_t = row['t']
            if prev_laser_state == 0 and row['laser_state']:
                # new laser pulse, reset cum
                cur_cum_this_trial = 0
                if row['ttm'] < 0:
                    cur_head_trial += 1
                    cur_state = 'head'
                elif row['ttm'] > 0:
                    cur_thorax_trial += 1
                    cur_state = 'thorax'
            prev_laser_state = row['laser_state']
            if row['zx'] > 0:
                cur_cum_this_trial = 1
            #gtdf['cum_wei_this_trial'] = cur_cum_this_trial
            row['cum_wei_this_trial'] = cur_cum_this_trial
            if cur_state=='head':
                row['head_trial'] = cur_head_trial
            elif cur_state=='thorax':
                row['thorax_trial'] = cur_thorax_trial
            gtdf.iloc[rowi] = row

    fig_cum = plt.figure('cum indx: %s'%label)
    ax_cum = fig_cum.add_subplot(111)
    pulse_nums = [1,2,3]
    this_data = {}
    head_data = []
    thorax_data = []

    sdx_e = []
    sdx_g = []
    sdx_c = []

    for pulse_num in pulse_nums:
        head_pulse_df = gtdf[ gtdf['head_trial']==pulse_num ]
        thorax_pulse_df = gtdf[ gtdf['thorax_trial']==pulse_num ]

        h = _do_group( head_pulse_df )
        t = _do_group( thorax_pulse_df )
        sdx_e, sdx_g, sdx_c = _update_latencies( head_pulse_df,sdx_e, sdx_g, sdx_c,'head')
        sdx_e, sdx_g, sdx_c = _update_latencies( thorax_pulse_df,sdx_e, sdx_g, sdx_c,'thorax')
        this_data['head%d'%pulse_num] = h
        this_data['thorax%d'%pulse_num] = t
        head_data.append( h )
        thorax_data.append( t )
        plot_cum( ax_cum, h, #label='head %d'%pulse_num,
                  lw=0.5, color=COLORS[HEAD])
        plot_cum( ax_cum, t, #label='thorax %d'%pulse_num,
                  lw=0.5, color=COLORS[THORAX])
    all_head_data = combine_cum_data( head_data )
    all_thorax_data = combine_cum_data( thorax_data )
    plot_cum( ax_cum, all_head_data, label='head',
              lw=2, color=COLORS[HEAD])
    plot_cum( ax_cum, all_thorax_data, label='thorax',
              lw=2, color=COLORS[THORAX])
    ax_cum.legend()

    sdx_e = np.array(sdx_e)
    sdx_g = np.array(sdx_g)
    sdx_c = np.array(sdx_c)

    buf = ''
    alpha = 0.95
    try:
        S,P,T = pairwise_logrank_test( sdx_e, sdx_g, sdx_c, alpha=alpha )
    except np.linalg.linalg.LinAlgError:
        buf += 'numerical errors computing logrank test'
    else:
        buf += '<h1>%s</h1>\n'%label
        buf += '<h2>pairwise logrank test</h2>\n'
        buf += '  analyses done using the <a href="http://lifelines.readthedocs.org">lifelines</a> library\n'
        buf += P._repr_html_()
        buf += '<h3>significant at alpha=%s?</h3>\n'%alpha
        buf += T._repr_html_()

    n_head   = len([x for x in sdx_g if x=='head'])
    n_thorax = len([x for x in sdx_g if x=='thorax'])
    p_value = P['head']['thorax']

    return {'fig':fig_cum,
            'ax':ax_cum,
            'n_head':n_head,
            'n_thorax':n_thorax,
            'p_value':p_value,
            'buf':buf,
            }

def plot_data(path, data):
    ci_html_buf = ''

    for exp_name in data:
        gts = data[exp_name].keys()

        laser = '130ht'
        gts_string = 'vs'.join(gts)
        figname = laser + '_' + gts_string

        fig = plt.figure("Song (%s)" % figname)
        ax = fig.add_subplot(1,1,1)

        datasets = {}
        for gt in gts:

            if flymad_analysis.genotype_is_exp(gt):
                color = flymad_plot.RED
                order = 1
            elif flymad_analysis.genotype_is_ctrl(gt):
                color = flymad_plot.BLACK
                order = 2
            elif flymad_analysis.genotype_is_trp_ctrl(gt):
                order = 3
                color = flymad_plot.BLUE
            else:
                color = 'cyan'
                order = 0

            gtdf = data[exp_name][gt]
            datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                                value=gtdf['mean']['zx'].values,
                                std=gtdf['std']['zx'].values,
                                n=gtdf['n']['zx'].values,
                                order=order,
                                df=gtdf,
                                label=flymad_analysis.human_label(gt),
                                color=color,
                                N=len(gtdf['df']['obj_id'].unique()))

        pvalue_buf = ''

        for gt in datasets:
            label=flymad_analysis.human_label(gt)
            if '>' not in label:
                continue

            # OK, this is a Gal4 + UAS - do head vs thorax stats
            gtdf = data[exp_name][gt]['df']
            ci_data = do_cum_incidence(gtdf, label)
            ci_html_buf += ci_data['buf']

            ax_cum = ci_data['ax']
            spine_placer(ax_cum, location='left,bottom' )

            ax_cum.set_ylabel('Fraction extending wing (%)')
            ax_cum.set_xlabel('Time since pulse onset (s)')

            note = '%s\n%s\np-value: %.3g\n%d flies\nn=%d, %d'%(label,
                                                                p2stars(ci_data['p_value']),
                                                            ci_data['p_value'],
                                                            len(gtdf['obj_id'].unique()),
                                                            ci_data['n_head'],
                                                            ci_data['n_thorax'])
            ax_cum.text(0, 1, #top left
                        note,
                        fontsize=10,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax_cum.transAxes,
                        zorder=0)

            ci_data['fig'].savefig(flymad_plot.get_plotpath(path,"song_cum_inc_%s.png" % figname),
                                    bbox_inches='tight')
            ci_data['fig'].savefig(flymad_plot.get_plotpath(path,"song_cum_inc_%s.svg" % figname),
                                    bbox_inches='tight')


#            for i in range(len(head_times)):
#                head_values = gtdf[gtdf['t']==head_times[i]]
#                thorax_values = gtdf[gtdf['t']==thorax_times[i]]
#                test1 = head_values['zx'].values
#                test2 = thorax_values['zx'].values
#                hval, pval = kruskal(test1, test2)
#                pvalue_buf += 'Pulse %d: Head vs thorax WEI p-value: %.3g (n=%d, %d)\n'%(
#                    i+1, pval, len(test1), len(test2) )

        #all experiments used identical activation times
        headtargetbetween = dict(xaxis=data['pIP10']['wtrpmyc']['first']['t'].values,
                                 where=data['pIP10']['wtrpmyc']['first']['ttm'].values > 0)
        thoraxtargetbetween = dict(xaxis=data['pIP10']['wtrpmyc']['first']['t'].values,
                                   where=data['pIP10']['wtrpmyc']['first']['ttm'].values < 0)

        flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=[headtargetbetween,thoraxtargetbetween],
                    sem=True,
                                                    note=pvalue_buf,
                    **datasets
        )

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wing extension index')
        ax.set_ylim([-0.05,0.4] if gts_string == "40347vswtrpmycvs40347trpmyc" else [-0.05,1.0])
        ax.set_xlim([-10,180])

        flymad_plot.retick_relabel_axis(ax, [0, 60, 120, 180],
                [0, 0.2, 0.4] if gts_string == "40347vswtrpmycvs40347trpmyc" else [0, 0.5, 1])

        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.png" % figname), bbox_inches='tight')
        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.svg" % figname), bbox_inches='tight')

    with open(flymad_plot.get_plotpath(path,"song_cum_in.html"),mode='w') as fd:
        fd.write(ci_html_buf)

if __name__ == "__main__":
    EXPS = {
        'P1':   {'exp':['wGP','G323'],
                 'ctrl':['wtrpmyc']},
        'pIP10':{'exp':['40347trpmyc','40347'],
                 'ctrl':['wtrpmyc']},
        'vPR6': {'exp':['5534trpmyc','5534'],
                 'ctrl':['wtrpmyc']},
        'vMS11':{'exp':['43702trp','43702'],
                 'ctrl':['wtrp']},
        'dPR1': {'exp':['41688trp','41688'],
                 'ctrl':['wtrp']},
    }

    CTRLS = ['wtrp','wtrpmyc']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    bin_size = '1S'
    cache_fname = os.path.join(path,'song_ctrls_%s.madplot-cache' % bin_size)
    cache_args = os.path.join(path, 'TRP_ctrls'), CTRLS, bin_size
    cdata = None
    if args.only_plot:
        cdata = madplot.load_bagfile_cache(cache_args, cache_fname)
    if cdata is None:
        cdata = prepare_data(os.path.join(path, 'TRP_ctrls'), bin_size, CTRLS)
        madplot.save_bagfile_cache(cdata, cache_args, cache_fname)

    cache_fname = os.path.join(path,'song_%s.madplot-cache' % bin_size)
    cache_args = path, EXPS, bin_size
    data = None
    if args.only_plot:
        data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if data is None:
        data = {}
        for exp_name in EXPS:
            data[exp_name] = prepare_data(os.path.join(path, exp_name), bin_size, EXPS[exp_name]['exp'])
        madplot.save_bagfile_cache(data, cache_args, cache_fname)

    #share the controls between experiments
    for exp_name in data:
        for ctrl_name in cdata:
            if ctrl_name in EXPS[exp_name]['ctrl']:
                data[exp_name][ctrl_name] = cdata[ctrl_name]
    for exp_name in data:
        gts = data[exp_name].keys()

        fname_prefix = flymad_plot.get_plotpath(path,'csv_song_exp_%s'%exp_name)
#        madplot.view_pairwise_stats_plotly(data[exp_name], gts,
#                                           fname_prefix,
#                                           align_colname='t',
#                                           stat_colname='zx',
#                                           layout_title='pvalues: WEI %s'%exp_name,
#                                           )
    plot_data(path, data)

#    plot_data(path, args.laser, gts, dfs)

    if args.show:
        plt.show()

