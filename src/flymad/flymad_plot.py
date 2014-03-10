import os.path

import strawlab_mpl.defaults as smd
from strawlab_mpl.many_timeseries import ManyTimeseries
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

import math
import numpy as np
import scipy.signal

BLACK       = '#292724'
DARK_GRAY   = '#939598'
LIGHT_GRAY  = '#e7e8e8'
#colors from colorbrewer2.org
RED         = '#D7191C'
ORANGE      = '#FDAE61'
BLUE        = '#0571B0'
GREEN       = '#1A9641'

LIGHT_BLUE  = '#92C5DE'
LIGHT_GREEN = '#A6D96A'

TS_DEFAULTS = {
    'many': dict(lw=0.2, color='k', alpha=0.6 ),
    'spread': dict(alpha=0.4, facecolor='red', edgecolor='none'),
    'value': dict(lw=2, color='red' ),
    'global': dict(rasterized=True),
}

def setup_defaults():
    rcParams = matplotlib.rcParams

    rcParams['legend.numpoints'] = 1
    rcParams['legend.fontsize'] = 'medium' #same as axis
    rcParams['legend.frameon'] = False
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1

def get_plotpath(path, name):
    path_out = os.path.join(os.path.dirname(path),'plots')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fig_out = os.path.join(path_out,name)
    print "wrote", fig_out
    return fig_out

def plot_timeseries_with_activation(ax, targetbetween=None, downsample=1, sem=False, **datasets):
    DEFAULT_COLORS = {"exp":RED,"ctrl":BLACK}

    def _ds(a):
        if downsample == 1:
            return a
        else:
            tmp = []
            for i in range(0,len(a),downsample):
                vals = a[i:i+downsample]
                tmp.append( np.mean(vals) )
            return np.array(tmp)

    if targetbetween is not None:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(targetbetween['xaxis'], 0, 1, where=targetbetween['where'],
                        edgecolor='none',
                        facecolor='Yellow', alpha=0.15, transform=trans,
                        zorder=1)

    #zorder = 1 = back
    top_zorder = 60
    bottom_zorder = 30

    cur_zorder = 2
    for data in sorted(datasets):
        exp = datasets[data]

        if exp.get('ontop'):
            this_zorder = top_zorder + cur_zorder
        else:
            this_zorder = bottom_zorder + cur_zorder

        print "plotting", data, "zorder", this_zorder

        if sem:
            spread = exp['std'] / np.sqrt(exp['n'])
        else:
            spread = exp['std']

        ax.fill_between(exp['xaxis'][::downsample], _ds(exp['value']+spread), _ds(exp['value']-spread),
                    alpha=0.1, color=exp.get('color',DEFAULT_COLORS.get(data,'k')),
                    zorder=this_zorder)

        ax.plot(exp['xaxis'][::downsample], _ds(exp['value']),
                    color=exp.get('color',DEFAULT_COLORS.get(data,'k')),label=exp.get('label',data),lw=2,
                    zorder=this_zorder+1)

        cur_zorder -= 2

    spine_placer(ax, location='left,bottom' )

    l = ax.legend(loc='upper right')
    l.set_zorder(1+top_zorder+cur_zorder)

#setup default plotting styles
smd.setup_defaults()
setup_defaults()

