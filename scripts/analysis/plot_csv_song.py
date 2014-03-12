import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import glob
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg

from scipy.stats import ttest_ind

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
import madplot

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0", "0.12.0")

HEAD    = +100
THORAX  = -100
OFF     = 0

def prepare_data(path, gts):

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

        #don't do any alignment other than starting time at zero. other
        df['t'] = df['t'].values - df['t'].values[0]

        #bin to  5 second bins:
        #FIXME: this is depressing dan code, lets just set a datetime index and resample properly...
        #df = df.resample('5S')
        df['t'] = df['t'] /5
        df['t'] = df['t'].astype(int)
        df['t'] = df['t'].astype(float)
        df['t'] = df['t'] *5
        df = df.groupby(df['t'], axis=0).mean() 

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

def run_stats (path, dfs):

    (expdf, expmean, expstd, expn,
            ctrldf, ctrlmean, ctrlstd, ctrln,
            ctrltrpdf, ctrltrpmean, ctrltrpstd, ctrltrpn,
            pooldf) = dfs
 
    print type(pooldf), pooldf.shape 
    p_values = pd.DataFrame()  
    df_ctrl = pooldf[pooldf['Genotype'] == ctrl_trp_genotype]
    df_exp1 = pooldf[pooldf['Genotype'] == EXP_GENOTYPE]
    df_exp2 = pooldf[pooldf['Genotype'] == EXP_GENOTYPE2]
    df_exp2['Genotype'] = 'VT40347GP'
    df_ctrl = df_ctrl[df_ctrl['t'] <= 485]
    df_exp1 = df_exp1[df_exp1['t'] <= 485]
    df_exp2 = df_exp2[df_exp2['t'] <= 485]
    df_ctrl = df_ctrl[df_ctrl['t'] >=-120]
    df_exp1 = df_exp1[df_exp1['t'] >=-120]
    df_exp2 = df_exp2[df_exp2['t'] >=-120]
    bins = np.linspace(-120,485,122)  # 5 second bins -120 to 485
    binned_ctrl = pd.cut(df_ctrl['t'], bins, labels= bins[:-1])
    binned_exp1 = pd.cut(df_exp1['t'], bins, labels= bins[:-1])
    binned_exp2 = pd.cut(df_exp2['t'], bins, labels= bins[:-1])
    for x in binned_ctrl.levels:               
        testctrl = df_ctrl['zx'][binned_ctrl == x]
        test1 = df_exp1['zx'][binned_exp1 == x]
        test2 = df_exp2['zx'][binned_exp2 == x]
        hval1, pval1 = ttest_ind(test1, testctrl)
        hval2, pval2 = ttest_ind(test2, testctrl) #too many identical values (zeros) in controls, so cannot do Kruskal.
        dftemp = pd.DataFrame({'Total_bins': binsize , 'Bin_number': x, 'P1': pval1, 'P2':pval2}, index=[x])
        p_values = pd.concat([p_values, dftemp])
    p_values1 = p_values[['Total_bins', 'Bin_number', 'P1']]
    p_values1.columns = ['Total_bins', 'Bin_number', 'P']
    p_values2 = p_values[['Total_bins', 'Bin_number', 'P2']]
    p_values2.columns = ['Total_bins', 'Bin_number', 'P']
    return p_values1, p_values2

def fit_to_curve ( p_values ):
    x = np.array(p_values['Bin_number'])
    logs = -1*(np.log(p_values['P']))
    y = np.array(logs)
    order = 6 #DEFINE ORDER OF POLYNOMIAL HERE.
    poly_params = np.polyfit(x,y,order)
    polynom = np.poly1d(poly_params)
    xPoly = np.linspace(min(x), max(x), 100)
    yPoly = polynom(xPoly)
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ax.plot(x, y, 'o', xPoly, yPoly, '-g')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('-log(p)')
    #plt.axhline(y=1.30103, color='k-')
    print polynom #lazy dan can't use python to solve polynomial eqns. boo.
    return (x, y, xPoly, yPoly, polynom)


def plot_data(path, data):

    for exp_name in data:
        gts = data[exp_name].keys()

        laser = '130ht'
        figname = laser + '_' + 'vs'.join(gts)

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
                                label=flymad_analysis.human_label(gt),
                                color=color,
                                N=len(gtdf['df']['obj_id'].unique()))

        #all experiments used identical activation times
        headtargetbetween = dict(xaxis=data['pIP10']['wtrpmyc']['first']['t'].values,
                                 where=data['pIP10']['wtrpmyc']['first']['ttm'].values > 0)
        thoraxtargetbetween = dict(xaxis=data['pIP10']['wtrpmyc']['first']['t'].values,
                                   where=data['pIP10']['wtrpmyc']['first']['ttm'].values < 0)

        flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=[headtargetbetween,thoraxtargetbetween],
                    sem=True,
                    **datasets
        )

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wing Ext. Index')
        ax.set_ylim([-0.05,1.0])
        ax.set_xlim([0,210])

        flymad_plot.retick_relabel_axis(ax, [20, 30, 50, 200], [0, 0.5, 1])

        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.png" % figname), bbox_inches='tight')
        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.svg" % figname), bbox_inches='tight')

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

    cache_fname = os.path.join(path,'song_ctrls.madplot-cache')
    cache_args = (os.path.join(path, 'TRP_ctrls'), CTRLS)
    cdata = None
    if args.only_plot:
        cdata = madplot.load_bagfile_cache(cache_args, cache_fname)
    if cdata is None:
        cdata = prepare_data(os.path.join(path, 'TRP_ctrls'), CTRLS)
        madplot.save_bagfile_cache(cdata, cache_args, cache_fname)

    cache_fname = os.path.join(path,'song.madplot-cache')
    cache_args = (path, EXPS)
    data = None
    if args.only_plot:
        data = madplot.load_bagfile_cache(cache_args, cache_fname)
    if data is None:
        data = {}
        for exp_name in EXPS:
            data[exp_name] = prepare_data(os.path.join(path, exp_name), EXPS[exp_name]['exp'])
        madplot.save_bagfile_cache(data, cache_args, cache_fname)

    #share the controls between experiments
    for exp_name in data:
        for ctrl_name in cdata:
            if ctrl_name in EXPS[exp_name]['ctrl']:
                data[exp_name][ctrl_name] = cdata[ctrl_name]

    plot_data(path, data)

#    #p_values1, p_values2 = run_stats(path, dfs)
#    #fit_to_curve( p_values1 )
#    #fit_to_curve( p_values2 )
#    plot_data(path, args.laser, gts, dfs)

    if args.show:
        plt.show()

