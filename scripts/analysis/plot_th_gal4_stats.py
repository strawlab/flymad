#!/usr/bin/env python
import pandas as pd
import scipy.stats
import datetime
import numpy as np
import strawlab_mpl.defaults as smd
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds
import matplotlib
from pairs2groups import label_homogeneous_groups, label_homogeneous_groups_pandas # github.com/astraw/pairs2groups
import argparse
import lifelines # http://lifelines.readthedocs.org
from lifelines.statistics import logrank_test, pairwise_logrank_test

import pandas as pd
import sys, os,re
import glob
import numpy as np
import matplotlib.pyplot as plt

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot

np.random.seed(3) # prevent plots from changing
MAX_LATENCY=20.0 # did not score longer than this...
EXTRA=5.0

class DataFrameHolder(object):
    def __init__(self,df):
        self.dfs = df
    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        k=self.dfs.keys()
        k.sort()
        return '\n'.join([ '<h4>%s</h4>\n'%ki +  self.dfs[ki]._repr_html_() for ki in k]   )

def gi2df(gi):
    """make a group_info dict into a pretty pandas DataFrame"""
    data = {}
    for name, row in zip(gi['group_names'], gi['p_values']):
        data[name] = row
    r =pd.DataFrame(data=data,index=gi['group_names'])
    return r

def gi2df2(gi):
    """make a group_info dict into a pretty pandas DataFrame"""
    data = {}
    for name, num_samples in zip(gi['group_names'], gi['num_samples']):
        data[name] = [num_samples]
    r =pd.DataFrame(data=data,index=['n'])
    return r

def plot_ci_for_ms(path, figname, vals, figsize=(6,4)):

    figname = os.path.splitext(figname)[0]

    NAMES = {'TH>trpA1 head':'TH>trpA1 (head)',
              'TH>trpA1 thorax':'TH>trpA1 (thorax)',
              'TH head':'TH (head)',
              'TH thorax':'TH (thorax)',
              'trpA1 head':'trpA1 (head)',
              'trpA1 thorax':'trpA1 (thorax)',
              'controls head':'controls (head)',
              'controls thorax':'controls (thorax)',
    }
    COLORS = {'TH>trpA1 head':flymad_plot.RED,
              'TH>trpA1 thorax':flymad_plot.ORANGE,
              'TH head':flymad_plot.GREEN,
              'TH thorax':flymad_plot.BLUE,
              'trpA1 head':flymad_plot.GREEN,
              'trpA1 thorax':flymad_plot.BLUE,
              'controls head':flymad_plot.GREEN,
              'controls thorax':flymad_plot.BLUE,
    }

    ORDER = ['TH>trpA1 head',
              'TH>trpA1 thorax',
              'TH head',
              'TH thorax',
              'trpA1 head',
              'trpA1 thorax',
              'controls head',
              'controls thorax',
    ]


    figure_title = "THGAL4 %s cumulative incidence" % figname
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    for gt in sorted(vals, cmp=lambda a,b: cmp(ORDER.index(a), ORDER.index(b))):
        ax.plot(vals[gt]['x'],vals[gt]['y'],
                lw=2,clip_on=False,
                color=COLORS[gt],label=NAMES[gt])

    spine_placer(ax, location='left,bottom' )
    ax.legend()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative incidence (%)')
    ax.set_ylim([0,100])
    ax.set_xlim([0,20])

    flymad_plot.retick_relabel_axis(ax, [0,10,20], [0,100])

    fig.savefig(flymad_plot.get_plotpath(path,"thgal4_ci_%s.png" % figname), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"thgal4_ci_%s.svg" % figname), bbox_inches='tight')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False,
                        default='.',
                        help='path to .df files')
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()
    dirname = args.path

    measurements = ['proboscis','wing','abdomen','jump']

    x_vals = {'TH>trpA1 head':0,
              'TH>trpA1 thorax':1,
              'TH head':2,
              'TH thorax':3,
              'trpA1 head':4,
              'trpA1 thorax':5,
              'controls head':2,
              'controls thorax':3,
              }

    buf = '<html><head>\n'
#    buf += '  <link href="http://netdna.bootstrapcdn.com/bootstrap/3.0.3/css/bootstrap.min.css" rel="stylesheet">\n'
    buf += '</head>\n'
    buf += '<body>Page generated %s\n'%( datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
#    for pooled_controls in [True]:
    for pooled_controls in [True,False]:
        if pooled_controls:
            pooled_str = 'pooled'
        else:
            pooled_str = 'notpooled'

        for measurement in measurements:
            fname = os.path.join(dirname,'%s_%s.df'%(measurement,pooled_str))
            df_all = pd.read_pickle(fname)

            #group_info._repr_html_()

            # --- plots --------------------------------
            fig1 = plt.figure('bar '+measurement+pooled_str,figsize=(3,5))
            ax = fig1.add_subplot(111)
            tick_labels = []
            xticks = []
            for name_key, group1 in df_all.groupby("name_key",sort=True):
                this_x_value = x_vals[name_key]
                this_y_values = group1['latency'].values
                this_y_values = np.clip(this_y_values,0,MAX_LATENCY)
                this_x_values = np.array([this_x_value]*len(this_y_values))
                uw = 0.2
                this_x_values = np.array(this_x_values) + np.random.uniform(-uw, uw, size=(len(this_y_values),))
                bar_width = 0.6
                ax.bar( [this_x_value-bar_width*0.5],
                        [ np.mean(this_y_values) ],
                        [ bar_width ],
                        clip_on=False,
                        color='black',
                        linewidth=0.5,
                        )
                ax.errorbar( [this_x_value],
                             [ np.mean(this_y_values) ],
                             [ scipy.stats.sem( this_y_values ) ],
                             clip_on=False,
                             color='black',
                             linewidth=0.5,
                             capsize=2,
                             )

                xticks.append( this_x_value )
                tick_labels.append( name_key )
            spine_placer(ax, location='left,bottom' )
            ax.spines['bottom'].set_color('none')
            ax.spines['bottom'].set_position(('outward',5))
            ax.set_yticks([0,20])
            ax.set_xticks(xticks)
            ax.set_xticklabels(tick_labels, rotation='vertical')
            ax.set_ylabel('latency (sec)')
            fig1.subplots_adjust(left=0.4,bottom=0.65) # do not clip text
            svg1_fname = 'th_gal4_latency_%s_%s.svg'%(measurement,pooled_str)
            fig1.savefig(svg1_fname)

            # --- plots --------------------------------
            fig2 = plt.figure('scatter '+measurement+pooled_str,figsize=(3,5))
            ax = fig2.add_subplot(111)
            tick_labels = []
            xticks = []

            for name_key, group1 in df_all.groupby("name_key",sort=True):

                this_x_value = x_vals[name_key]
                this_y_values = group1['latency'].values
                this_y_values = np.clip(this_y_values,0,MAX_LATENCY)
                this_y_values[ this_y_values==MAX_LATENCY ] = MAX_LATENCY+EXTRA

                this_x_values = np.array([this_x_value]*len(this_y_values))
                uw = 0.2
                this_x_values = np.array(this_x_values) + np.random.uniform(-uw, uw, size=(len(this_y_values),))

                ax.plot( this_x_values, this_y_values, 'ko',
                         mew=0.3,
                         mfc='none',
                         ms=3.0,
                         clip_on=False )

                xticks.append( this_x_value )
                tick_labels.append( name_key )
            spine_placer(ax, location='left,bottom' )
            ax.spines['bottom'].set_color('none')
            ax.spines['bottom'].set_position(('outward',5))
            ax.spines['left'].set_bounds(0,MAX_LATENCY)
            ax.set_yticks([0,MAX_LATENCY,MAX_LATENCY+EXTRA])
            ax.set_yticklabels(['0','20','$\phi$']); assert MAX_LATENCY==20
            ax.set_xticks(xticks)
            ax.set_xticklabels(tick_labels, rotation='vertical')
            ax.set_ylabel('latency (sec)')
            fig2.subplots_adjust(left=0.4,bottom=0.65) # do not clip text
            svg2_fname = 'th_gal4_latency_scatter_%s_%s.svg'%(measurement,pooled_str)
            fig2.savefig(svg2_fname)

            # --- begin making web page

            buf += '<h1>' + measurement + ' ' + pooled_str + '</h1>\n'
            buf += '<h2>plots</h2>\n'
            buf += '<object type="image/svg+xml" data="%s">Your browser does not support SVG</object>'%(svg1_fname,)
            buf += '<object type="image/svg+xml" data="%s">Your browser does not support SVG</object>'%(svg2_fname,)

            # --- cumulative incidence plots ----------
            ms_data = {}

            fig3 = plt.figure('cum incidence '+measurement+pooled_str,figsize=(3,5))
            ax = fig3.add_subplot(111)
            for name_key, group1 in df_all.groupby("name_key",sort=True):
                frac = 1.0/len( group1 )

                latencies = list(group1['latency'].values)
                latencies.sort()
                this_y_vals = [0]
                this_x_vals = [0]
                maxy = 0
                for latency in latencies:
                    if latency >= MAX_LATENCY:
                        break
                    this_x_vals.append( latency )
                    this_y_vals.append( maxy*100.0 )
                    maxy += frac
                    this_x_vals.append( latency )
                    this_y_vals.append( maxy*100.0 )

                this_x_vals.append( MAX_LATENCY )
                this_y_vals.append( maxy*100.0 )

                ax.plot( this_x_vals, this_y_vals, '-', label=name_key )

                ms_data[name_key] = dict(x=this_x_vals, y=this_y_vals)

            ax.legend()
            spine_placer(ax, location='left,bottom' )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cumulative incidence (%)')
            fig3.subplots_adjust(left=0.4,bottom=0.35) # do not clip text

            svg3_fname = 'th_gal4_latency_cuminc_%s_%s.svg'%(measurement,pooled_str)
            fig3.savefig(svg3_fname)
            buf += '<object type="image/svg+xml" data="%s">Your browser does not support SVG</object>'%(svg3_fname,)

            plot_ci_for_ms(dirname, '%s_%s' % (measurement,pooled_str), ms_data)

            # - survival fits ------------------------
            #  "survival functions" for right-censored events

            #kmf = lifelines.KaplanMeierFitter()

            sdx_e = []
            sdx_g = []
            sdx_c = []
            for name_key, group1 in df_all.groupby("name_key",sort=True):
                latency_arr = group1['latency'].values
                C = latency_arr < MAX_LATENCY
                #kmf.fit(latency_arr, censorship=C, label=name_key )
                for i in range(len(C)):
                    sdx_e.append( latency_arr[i] )
                    sdx_g.append( name_key )
                    sdx_c.append( C[i] )

            # --- stats --------------------------------

            sdx_e = np.array(sdx_e)
            sdx_g = np.array(sdx_g)
            sdx_c = np.array(sdx_c)

            alpha = 0.95
            try:
                S,P,T = pairwise_logrank_test( sdx_e, sdx_g, sdx_c, alpha=alpha )
            except np.linalg.linalg.LinAlgError:
                buf += 'numerical errors computing logrank test'
            else:
                buf += '<h2>pairwise logrank test</h2>\n'
                buf += '  analyses done using the <a href="http://lifelines.readthedocs.org">lifelines</a> library\n'
                buf += P._repr_html_()
                buf += '<h3>significant at alpha=%s?</h3>\n'%alpha
                buf += T._repr_html_()

            '''
            clipped = df_all.copy()
            clipped['latency'] = clipped['latency'].clip(upper=MAX_LATENCY)
            group_info = label_homogeneous_groups_pandas( clipped,
                                                          groupby_column_name='name_key',
                                                          value_column_name='latency')
            buf += '<h2>clipped comparisons</h2>\n'
            buf += gi2df2(group_info)._repr_html_()
            buf += gi2df(group_info)._repr_html_()


            valid_latencies_only = df_all.copy()
            valid_latencies_only = valid_latencies_only[ valid_latencies_only['latency'] < MAX_LATENCY ]
            group_info = label_homogeneous_groups_pandas( valid_latencies_only,
                                                          groupby_column_name='name_key',
                                                          value_column_name='latency')
            buf += '<h2>valid comparisons</h2>\n'
            buf += gi2df2(group_info)._repr_html_()
            buf += gi2df(group_info)._repr_html_()
            '''

    buf += '</body></html>'
    with open('th_gal4.html',mode='w') as fd:
        fd.write(buf)
