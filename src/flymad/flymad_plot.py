import os.path
import math
import md5

import numpy as np
import scipy.signal

import strawlab_mpl.defaults as smd
from strawlab_mpl.many_timeseries import ManyTimeseries
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec

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

EXP_COLORS = [RED, ORANGE, BLUE, GREEN, LIGHT_BLUE, LIGHT_GREEN]
CTRL_COLORS = [BLACK, DARK_GRAY]

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
    matplotlib.rc('font', size=8)

def get_plotpath(path, name):
    if os.path.isdir(path):
        default = path
    else:
        default = os.path.dirname(path)
    plotdir = os.environ.get('FLYMAD_PLOT_DIR', default)
    path_out = os.path.join(plotdir,'plots')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fig_out = os.path.join(path_out,name)
    print "wrote", fig_out
    return fig_out

def retick_relabel_axis(ax, xticks, yticks, xformat_func=None, yformat_func=None):
    #set the xlim and ylim before calling this function so it can tick them.
    #other ticks are placed at xticks and yticks via format func
    #
    #def format_func(tick):
    #    return "foo %s" % tick
    #
    if xformat_func is None:
        xformat_func = str
    if yformat_func is None:
        yformat_func = str

    #set to remove duplicates
    all_xticks = sorted(set(list(ax.get_xlim()) + xticks))
    all_yticks = sorted(set(list(ax.get_ylim()) + yticks))

    #defined labels
    xlbls = {i:xformat_func(i) for i in xticks}
    ylbls = {i:yformat_func(i) for i in yticks}

    #now remove labels on unlabeled ticks ('')
    if xlbls:
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([xlbls.get(i,'') for i in all_xticks]))
        ax.xaxis.set_major_locator(mticker.FixedLocator(all_xticks))
    if ylbls:
        ax.yaxis.set_major_formatter(mticker.FixedFormatter([ylbls.get(i,'') for i in all_yticks]))
        ax.yaxis.set_major_locator(mticker.FixedLocator(all_yticks))

def get_gridspec_to_fit_nplots(n):
    j = math.sqrt(n)
    nr = math.floor(j)
    nc = math.ceil(n / nr)
    return gridspec.GridSpec(int(nr),int(nc)), (nr, nc)

def plot_timeseries_with_activation(ax, targetbetween=None, downsample=1, sem=False,
                                    legend_location='upper right', note="",
                                    individual=None, individual_title=None,
                                    **datasets):

    ORDER_LAST = 100
    DEFAULT_COLORS = {"exp":RED,"ctrl":BLACK}

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    def _ds(a):
        if downsample == 1:
            return a
        else:
            tmp = []
            for i in range(0,len(a),downsample):
                vals = a[i:i+downsample]
                tmp.append( np.mean(vals) )
            return np.array(tmp)

    def _sort_by_order(a,b):
        return cmp(datasets[a].get('order', ORDER_LAST), datasets[b].get('order', ORDER_LAST))

    def _fill_between(f_ax, f_xaxis, f_where, f_facecolor):
        f_ax.fill_between(f_xaxis, 0, 1, where=f_where,
                          edgecolor='none', facecolor=f_facecolor,
                          alpha=0.15, transform=trans, zorder=1)


    if targetbetween is not None:
        if not (isinstance(targetbetween, list) or isinstance(targetbetween, tuple)):
            targetbetween = [targetbetween]
        for tb in targetbetween:
            _fill_between(ax, tb['xaxis'], tb['where'], tb.get('facecolor','yellow'))

    note += "+/- SEM" if sem else "+/- STD"
    note += "\n"
    note += "" if downsample == 1 else ("downsample x %d\n" % downsample)

    #zorder = 1 = back
    top_zorder = 60
    bottom_zorder = 30

    cur_zorder = 2
    for data in sorted(datasets.keys(), cmp=_sort_by_order):
        exp = datasets[data]

        label = exp.get('label',data)

        note += "N(%s)=%s\n" % (label,exp.get('N','??'))

        if exp.get('ontop'):
            this_zorder = top_zorder + cur_zorder
        else:
            this_zorder = bottom_zorder + cur_zorder

        print "plotting %s (%s) zorder %s" % (label,data,this_zorder)

        if sem:
            spread = exp['std'] / np.sqrt(exp['n'])
        else:
            spread = exp['std']

        ax.fill_between(exp['xaxis'][::downsample], _ds(exp['value']+spread), _ds(exp['value']-spread),
                    alpha=0.1, color=exp.get('color',DEFAULT_COLORS.get(data,'k')),
                    zorder=this_zorder)

        ax.plot(exp['xaxis'][::downsample], _ds(exp['value']),
                    color=exp.get('color',DEFAULT_COLORS.get(data,'k')),label=label,lw=2,
                    zorder=this_zorder+1)

        cur_zorder -= 2

    spine_placer(ax, location='left,bottom' )

    l = ax.legend(loc=legend_location)
    l.set_zorder(1+top_zorder+cur_zorder)

    ax.text(0, 1, #top left
            note,
            fontsize=10,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            zorder=0)

    axs = [ax]
    figs = {}
    if (individual is not None) and isinstance(individual, dict):
        for data in individual:
            try:
                #try to avoid creating many duplicate figures
                #based on the title be unique if provided
                if individual_title:
                    fignum = hash(individual_title)
                    if plt.fignum_exists(fignum):
                        continue
                else:
                    fignum = None
                    individual_title = ""

                individual_title += data

                gdf = datasets[data]['df']
                groupcol = individual[data]['groupby']
                xcol = individual[data]['xaxis']
                ycol = individual[data]['yaxis']

                grouper = gdf.groupby(groupcol)
                nts = len(grouper)

                gs,nr_nc = get_gridspec_to_fit_nplots(nts)

                fig2 = plt.figure(fignum, figsize=(2*max(nr_nc),2*max(nr_nc)))
                fig2.suptitle(individual_title)
                print 'plotting', nts, 'individual timeseries for', data

                for gridspec_id,(name,group) in zip(gs,grouper):
                    #we can't assume these are numpy arrays here, but
                    #np.array does the right thing and converts pandas things
                    #while being no-op on np arrays
                    grp_xaxis = np.array(group[xcol])
                    grp_yaxis = np.array(group[ycol])

                    iax = fig2.add_subplot(gridspec_id)
                    iax.plot(grp_xaxis[::downsample], _ds(grp_yaxis),
                             label=name, color='k')
                    iax.legend(loc=legend_location)
                    print "\tplot",name,xcol,"vs",ycol

#                    if fb_wherecol:
#                        _fill_between(iax,
#                                      grp_xaxis,
#                                      np.array(group[fb_wherecol]) > 0,
#                                      'yellow')

                    axs.append(iax)

                figs[md5.md5(individual_title).hexdigest()] = fig2

            except KeyError, e:
                print "\terror plotting individual timeseries (%s)" % e

    return l, axs, figs

#setup default plotting styles
smd.setup_defaults()
setup_defaults()

