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

def plot_timeseries_with_activation(ax, exp=dict(), ctrl=dict(), exp2=dict(), targetbetween=None, downsample=1, sem=False):
    def _ds(a):
        if downsample == 1:
            return a
        else:
            return scipy.signal.resample(a, (len(a)//downsample) + 1)

    #zorder = 1 = back
    #FIXME: make controls black, but the other black, not perfect black

    if ctrl:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        if targetbetween is not None:
            ax.fill_between(ctrl['xaxis'], 0, 1, where=targetbetween,
                            edgecolor='none',
                            facecolor='Yellow', alpha=0.15, transform=trans,
                            zorder=1)

    #plot the experiment on top of the control
    exp_ontop = exp.get('ontop',not ctrl.get('ontop', False))

    if exp_ontop:
        ctrl_zorder = 2
        exp_zorder = 6
    else:
        ctrl_zorder = 6
        exp_zorder = 2

    if exp:
        if sem:
            spread = exp['std'] / np.sqrt(exp['n'])
        else:
            spread = exp['std']

        ax.fill_between(exp['xaxis'][::downsample], _ds(exp['value']+spread), _ds(exp['value']-spread),
                    alpha=0.1, color=RED,
                    zorder=exp_zorder)

        ax.plot(exp['xaxis'][::downsample], _ds(exp['value']),
                    color=RED,label=exp.get('label'),lw=2,
                    zorder=exp_zorder+1)

    if ctrl:
        if sem:
            spread = ctrl['std'] / np.sqrt(ctrl['n'])
        else:
            spread = ctrl['std']


        ax.fill_between(ctrl['xaxis'][::downsample], _ds(ctrl['value']+spread), _ds(ctrl['value']-spread),
                    alpha=0.1, color=BLACK,
                    zorder=ctrl_zorder)
        ax.plot(ctrl['xaxis'][::downsample], _ds(ctrl['value']),
                    color=BLACK,label=ctrl.get('label'),lw=2,
                    zorder=ctrl_zorder+1)

    if exp2:
        if sem:
            spread = exp2['std'] / np.sqrt(exp2['n'])
        else:
            spread = exp2['std']

        exp_zorder = exp_zorder + 2

        ax.fill_between(exp2['xaxis'][::downsample], _ds(exp2['value']+spread), _ds(exp2['value']-spread),
                    alpha=0.1, color=BLUE,
                    zorder=ctrl_zorder)
        ax.plot(exp2['xaxis'][::downsample], _ds(exp2['value']),
                    color=BLUE,label=exp2.get('label'),lw=2,
                    zorder=ctrl_zorder+1)

    spine_placer(ax, location='left,bottom' )

    ax.legend(loc='upper right')

#setup default plotting styles
smd.setup_defaults()
setup_defaults()

