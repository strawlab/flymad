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

def setup_defaults():
    rcParams = matplotlib.rcParams

    rcParams['legend.numpoints'] = 1
    rcParams['legend.fontsize'] = 'medium' #same as axis
    rcParams['legend.frameon'] = False
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1
    matplotlib.rc('font', size=8)

smd.setup_defaults()
setup_defaults()

import pandas as pd
import sys, os,re
import glob
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3) # prevent plots from changing
MAX_LATENCY=10.0 # did not score longer than this...
EXTRA=2.0

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False,
                        default='.',
                        help='path to .df files')
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

    buf = '<html><body>Page generated %s'%( datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
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
            fig1 = plt.figure('bar '+measurement+pooled_str,figsize=(1.5,2))
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
            ax.set_yticks([0,10])
            ax.set_xticks(xticks)
            ax.set_xticklabels(tick_labels, rotation='vertical')
            ax.set_ylabel('latency (sec)')
            fig1.subplots_adjust(left=0.4,bottom=0.65) # do not clip text
            svg1_fname = 'th_gal4_latency_%s_%s.svg'%(measurement,pooled_str)
            fig1.savefig(svg1_fname)

            # --- plots --------------------------------
            fig2 = plt.figure('scatter '+measurement+pooled_str,figsize=(1.5,2))
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
            ax.set_yticklabels(['0','10','']); assert MAX_LATENCY==10
            ax.set_xticks(xticks)
            ax.set_xticklabels(tick_labels, rotation='vertical')
            ax.set_ylabel('latency (sec)')
            fig2.subplots_adjust(left=0.4,bottom=0.65) # do not clip text
            svg2_fname = 'th_gal4_latency_scatter_%s_%s.svg'%(measurement,pooled_str)
            fig2.savefig(svg2_fname)

            # --- stats --------------------------------
            clipped = df_all.copy()
            clipped['latency'] = clipped['latency'].clip(upper=MAX_LATENCY)
            group_info = label_homogeneous_groups_pandas( clipped,
                                                          groupby_column_name='name_key',
                                                          value_column_name='latency')
            buf += '<h1>' + measurement + ' ' + pooled_str + '</h1>\n'
            buf += '<object type="image/svg+xml" data="%s">Your browser does not support SVG</object>'%(svg1_fname,)
            buf += '<object type="image/svg+xml" data="%s">Your browser does not support SVG</object>'%(svg2_fname,)
            buf += gi2df(group_info)._repr_html_()

    buf += '</body></html>'
    with open('th_gal4.html',mode='w') as fd:
        fd.write(buf)
