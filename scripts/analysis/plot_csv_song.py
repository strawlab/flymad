import argparse
import glob
import os
import pickle

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

def prepare_data(path, gts):
    data = {}

    path_out = path + "/outputs/"
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    #PROCESS SCORE FILES:
    pooldf = pd.DataFrame()
    for df,metadata in flymad_analysis.courtship_combine_csvs_to_dataframe(path, as_is_laser_state=False):
        csvfilefn,experimentID,date,time,genotype,laser,repID = metadata
        #ALIGN BY LASER OFF TIME (=t0)
        if (df['laser_state']==1).any():
            lasermask = df[df['laser_state']==1]
            df['t'] = df['t'] - np.min(lasermask['t'].values)#laser off time =0               
        else:  
            print "No laser detected!!!!!!!!!! NOT PLOTTING THIS TRIAL"
            continue

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        #bin to  5 second bins:
        #FIXME: this is depressing dan code, lets just set a datetime index and resample properly...
        #df = df.resample('5S')
        df['t'] = df['t'] /5
        df['t'] = df['t'].astype(int)
        df['t'] = df['t'].astype(float)
        df['t'] = df['t'] *5
        df = df.groupby(df['t'], axis=0).mean() 

        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        pooldf = pd.concat([pooldf, df[['Genotype','lasergroup', 't','zx', 'laser_state']]])  

    data = {}
    for gt in gts:
        gtdf = pooldf[pooldf['Genotype'] == gt]
        print gt,gtdf

        assert len(gtdf['lasergroup'].unique()) == 1, "only one lasergroup handled, see --laser"

        data[gt] = dict(mean=gtdf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'laser_state']].mean().astype(float),
                        std=gtdf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'laser_state']].std().astype(float),
                        n=gtdf.groupby(['t'],  as_index=False)[['zx', 'dtarget', 'laser_state']].count().astype(float))

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

    COLORS = {'5534trpmyc':flymad_plot.RED,
              '5534':flymad_plot.BLACK,
              '40347trpmyc':flymad_plot.RED,
              '40347':flymad_plot.BLACK,
              'wGP':flymad_plot.RED,
              'G323':flymad_plot.BLACK,
              '43702trp':flymad_plot.RED,
              '43702':flymad_plot.BLACK,
              '41688trp':flymad_plot.RED,
              '41688':flymad_plot.BLACK,
              'wtrp':flymad_plot.GREEN,
              'wtrpmyc':flymad_plot.BLUE,
    }

    path_out = path + "/outputs/"

    for exp_name in data:
        gts = data[exp_name].keys()

        laser = '130ht'
        figname = laser + '_' + 'vs'.join(gts)

        fig = plt.figure("Song (%s)" % figname)
        ax = fig.add_subplot(1,1,1)

        datasets = {}
        for gt in gts:
            gtdf = data[exp_name][gt]
            datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                                value=gtdf['mean']['zx'].values,
                                std=gtdf['std']['zx'].values,
                                n=gtdf['n']['zx'].values,
                                color=COLORS[gt])

        #all experiments were identical, so take activation times from the last one
        targetbetween = dict(xaxis=data[exp_name]['wtrp']['mean']['t'].values,
                             where=data[exp_name]['wtrp']['mean']['laser_state'].values>0)

        flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=targetbetween,
                    sem=True,
                    **datasets
        )

        ax.set_xlabel('Time (s)')
        #ax.set_ylabel('Wing Ext. Index, +/- SEM')
        #ax.set_title('Wing Extension (%s)' % laser, size=12)
        ax.set_ylim([-0.1,1.0])
        #ax.set_xlim([-60,480])

        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.png" % figname), bbox_inches='tight')
        fig.savefig(flymad_plot.get_plotpath(path,"song_%s.svg" % figname), bbox_inches='tight')

if __name__ == "__main__":
    EXPS = {
        'dPR1':{'exp':'5534trpmyc',
               'ctrl':'5534'},
        'pIP10':{'exp':'40347trpmyc',
                 'ctrl':'40347'},
        'P1':{'exp':'wGP',
              'ctrl':'G323'},
        'vMS11':{'exp':'43702trp',
                 'ctrl':'43702'},
        'vPR6':{'exp':'41688trp',
                'ctrl':'41688'}
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
            data[exp_name] = prepare_data(os.path.join(path, exp_name), EXPS[exp_name].values())
        madplot.save_bagfile_cache(data, cache_args, cache_fname)

    #share the controls between experiments
    for exp_name in data:
        for ctrl_name in cdata:
            data[exp_name][ctrl_name] = cdata[ctrl_name]

    plot_data(path, data)

#    #p_values1, p_values2 = run_stats(path, dfs)
#    #fit_to_curve( p_values1 )
#    #fit_to_curve( p_values2 )
#    plot_data(path, args.laser, gts, dfs)

    if args.show:
        plt.show()

