import os.path

import strawlab_mpl.defaults as smd
from strawlab_mpl.many_timeseries import ManyTimeseries
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

import numpy as np

BLACK       = 'k'
DARK_GRAY   = '#939598'
LIGHT_GRAY  = '#e7e8e8'
#colors from ggplot
BLUE        = '#348ABD'
PURPLE      = '#7A68A6'
RED         = '#A60628'
GREEN       = '#467821'
MID_RED     = '#CF4457'
MID_BLUE    = '#188487'
ORANGE_RED  = '#E24A33'

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

def plot_timeseries_with_activation(ax, exp, ctrl, exp2=None, targetbetween=None, downsample=1, sem=False):
    #zorder = 1 = back
    #downsample ::1 is a noop
    ds = downsample

    #FIXME: make controls black, but the other black, not perfect black

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    if targetbetween is not None:
        ax.fill_between(ctrl['xaxis'][::ds], 0, 1, where=targetbetween[::ds],
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

    if sem:
        spread = exp['std'][::ds] / np.sqrt(exp['n'][::ds])
    else:
        spread = exp['std'][::ds]

    ax.fill_between(exp['xaxis'][::ds], exp['value'][::ds]+spread, exp['value'][::ds]-spread,
                alpha=0.1, color='b',
                zorder=exp_zorder)
    ax.plot(exp['xaxis'][::ds], exp['value'][::ds],
                color='b',label=exp.get('label'),
                zorder=exp_zorder+1)

    if sem:
        spread = ctrl['std'][::ds] / np.sqrt(ctrl['n'][::ds])
    else:
        spread = ctrl['std'][::ds]


    ax.fill_between(ctrl['xaxis'][::ds], ctrl['value'][::ds]+spread, ctrl['value'][::ds]-spread,
                alpha=0.1, color='r',
                zorder=ctrl_zorder)
    ax.plot(ctrl['xaxis'][::ds], ctrl['value'][::ds],
                color='r',label=ctrl.get('label'),
                zorder=ctrl_zorder+1)

    if exp2 is not None:
        if sem:
            spread = exp2['std'][::ds] / np.sqrt(exp2['n'][::ds])
        else:
            spread = exp2['std'][::ds]

        exp_zorder = exp_zorder + 2

        ax.fill_between(exp2['xaxis'][::ds], exp2['value'][::ds]+spread, exp2['value'][::ds]-spread,
                    alpha=0.1, color='g',
                    zorder=ctrl_zorder)
        ax.plot(exp2['xaxis'][::ds], exp2['value'][::ds],
                    color='g',label=exp2.get('label'),
                    zorder=ctrl_zorder+1)

    spine_placer(ax, location='left,bottom' )

    ax.legend(loc='upper right')

#setup default plotting styles
smd.setup_defaults()
setup_defaults()

