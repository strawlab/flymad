import argparse
import glob
import os
import subprocess
import cPickle as pickle

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

def _get_targets(path, date):
    #first we look for a png file corresponding to the scored MP4 (for
    #consistency with the initial submission)

    def _mp4_click(image_path, cache_path):
        img = mimg.imread(image_path)
        fig1 = plt.figure()
        fig1.set_size_inches(12,8)
        fig1.subplots_adjust(hspace=0)
        ax1 = fig1.add_subplot(1,1,1)	

        #the original wide field camera was 659x494px. The rendered mp4 is 384px high
        #the widefield image is padded with a 10px margin, so it is technically 514 high.
        #scaling h=384->514 means new w=1371
        #
        #the image origin is top-left because matplotlib
        ax1.imshow(img, extent=[0,1371,514,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax1.axis('off') 

        targets = []
        def _onclick(target):
            #subtract 10px for the margin
            xydict = {'x': target.xdata-10, 'y': target.ydata-10}
            targets.append(xydict)

        cid = fig1.canvas.mpl_connect('button_press_event', _onclick)
        plt.show()
        fig1.canvas.mpl_disconnect(cid)

        with open(cache_path, 'wb') as f:
            pickle.dump(targets, f, -1)

        return targets

    #cached results
    mp4pngcache = glob.glob(os.path.join(path,'*%s*.mp4.png.madplot-cache' % date))
    if len(mp4pngcache) == 1:
        return pickle.load( open(mp4pngcache[0],'rb') )

    mp4pngcache = os.path.join(path,'*%s*.mp4.png.madplot-cache' % date)
    
    mp4png = glob.glob(os.path.join(path,'*%s*.mp4.png' % date))
    if len(mp4png) == 1:
        return _mp4_click(mp4png[0], mp4png[0] + '.madplot-cache')

    mp4 = glob.glob(os.path.join(path,'*%s*.mp4' % date))
    if len(mp4) == 1:
        mp4 = mp4[0]
        mp4png = mp4 + '.png'
        #make a thumbnail
        subprocess.check_call("ffmpeg -i %s -vframes 1 -an -f image2 -y %s" % (mp4,mp4png),
                              shell=True)
        return _mp4_click(mp4png, mp4png + '.madplot-cache')

    return []

def prepare_data(path, only_laser, gts):
    data = {}

    #PROCESS SCORE FILES:
    pooldf = pd.DataFrame()
    for df,metadata in flymad_analysis.courtship_combine_csvs_to_dataframe(path, as_is_laser_state=False):
        csvfilefn,experimentID,date,time,genotype,laser,repID = metadata
        if laser != only_laser:
            print "\tskipping laser", laser
            continue

        if genotype not in gts:
            print "\tskipping genotype", genotype
            continue

        targets = _get_targets(path, date)
        assert len(targets) == 4
        targets = pd.DataFrame(targets)
        targets = (targets + 0.5).astype(int)

        #CALCULATE DISTANCE FROM TARGETs, KEEP MINIMUM AS dtarget
        if targets is not None:
            dist = pd.DataFrame.copy(df, deep=True)
            dist['x0'] = df['x'] - targets.ix[0,'x']
            dist['y0'] = df['y'] - targets.ix[0,'y']
            dist['x1'] = df['x'] - targets.ix[1,'x']
            dist['y1'] = df['y'] - targets.ix[1,'y']
            dist['x2'] = df['x'] - targets.ix[2,'x']
            dist['y2'] = df['y'] - targets.ix[2,'y']
            dist['x3'] = df['x'] - targets.ix[3,'x']
            dist['y3'] = df['y'] - targets.ix[3,'y']
            dist['d0'] = ((dist['x0'])**2 + (dist['y0'])**2)**0.5
            dist['d1'] = ((dist['x1'])**2 + (dist['y1'])**2)**0.5
            dist['d2'] = ((dist['x2'])**2 + (dist['y2'])**2)**0.5
            dist['d3'] = ((dist['x3'])**2 + (dist['y3'])**2)**0.5
            df['dtarget'] = dist.ix[:,'d0':'d3'].min(axis=1)               
        else:
            df['dtarget'] = 0

        #ALIGN BY LASER OFF TIME (=t0)
        if (df['laser_state']==1).any():
            lasermask = df[df['laser_state']==1]
            df['t'] = df['t'] - np.max(lasermask['t'].values)#laser off time =0               
        else:  
            print "No laser detected!!!!!!!!!! NOT PLOTTING THIS TRIAL"
            continue

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
            exp2df, exp2mean, exp2std, exp2n,
            ctrldf, ctrlmean, ctrlstd, ctrln,
            pooldf) = dfs
 
    print type(pooldf), pooldf.shape 
    p_values = pd.DataFrame()  
    df_ctrl = pooldf[pooldf['Genotype'] == CTRL_GENOTYPE]
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

def plot_data(path, laser, dfs):

    COLORS = {'wtrpmyc':flymad_plot.BLACK,
              'wGP':flymad_plot.RED,
              '40347trpmyc':flymad_plot.ORANGE,
              'G323':flymad_plot.BLUE,
              '40347':flymad_plot.GREEN}

    figname = laser + '_' + '_'.join(dfs)

    datasets = {}
    for gt in dfs:
        if flymad_analysis.genotype_is_exp(gt):
            order = 1
        elif flymad_analysis.genotype_is_ctrl(gt):
            order = 2
        else:
            order = 3
        gtdf = dfs[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['zx'].values,
                            std=gtdf['std']['zx'].values,
                            n=gtdf['n']['zx'].values,
                            label=flymad_analysis.human_label(gt),
                            order=order,
                            color=COLORS[gt],
                            N=len(gtdf['df']['obj_id'].unique()))
    ctrlmean = dfs['wtrpmyc']['mean']

    fig = plt.figure("Courtship Wingext 10min (%s)" % laser)
    ax = fig.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=dict(xaxis=ctrlmean['t'].values,
                                       where=ctrlmean['laser_state'].values>0),
                    sem=True,
                    note="laser %s\n" % laser,
                    **datasets
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wing Ext. Index')
    ax.set_ylim([-0.1,0.6])
    ax.set_xlim([-60,480])

    fig.savefig(flymad_plot.get_plotpath(path,"following_and_WingExt_%s.png" % figname), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"following_and_WingExt_%s.svg" % figname), bbox_inches='tight')

    datasets = {}
    for gt in dfs:
        if flymad_analysis.genotype_is_exp(gt):
            order = 1
        elif flymad_analysis.genotype_is_ctrl(gt):
            order = 2
        else:
            order = 3
        gtdf = dfs[gt]
        datasets[gt] = dict(xaxis=gtdf['mean']['t'].values,
                            value=gtdf['mean']['dtarget'].values,
                            std=gtdf['std']['dtarget'].values,
                            n=gtdf['n']['dtarget'].values,
                            label=flymad_analysis.human_label(gt),
                            order=order,
                            color=COLORS[gt],
                            N=len(gtdf['df']['obj_id'].unique()))
    ctrlmean = dfs['wtrpmyc']['mean']

    fig = plt.figure("Courtship Dtarget 10min (%s)" % laser)
    ax = fig.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    targetbetween=dict(xaxis=ctrlmean['t'].values,
                                       where=ctrlmean['laser_state'].values>0),
                    sem=True,
                    legend_location='lower right',
                    note="laser %s\n" % laser,
                    **datasets
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (mm)')
    #ax.set_ylim([20,120])
    ax.set_xlim([-60,480])

    fig.savefig(flymad_plot.get_plotpath(path,"following_and_dtarget_%s.png" % figname), bbox_inches='tight')
    fig.savefig(flymad_plot.get_plotpath(path,"following_and_dtarget_%s.png" % figname), bbox_inches='tight')

if __name__ == "__main__":
    CTRL_GENOTYPE = 'wtrpmyc'
    EXP_GENOTYPE = 'wGP'
    EXP2_GENOTYPE = '40347trpmyc'
    CTRL2_GENOTYPE = 'G323'
    CTRL3_GENOTYPE = '40347'

    gts = EXP_GENOTYPE, EXP2_GENOTYPE, CTRL_GENOTYPE, CTRL2_GENOTYPE, CTRL3_GENOTYPE

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--laser', default='140hpc', help='laser specifier')
    parser.add_argument('--dose-response', action='store_true', default=False,
                        help='plot dose response')

    args = parser.parse_args()
    path = args.path[0]

    cache_fname = os.path.join(path,'courtship.madplot-cache')
    cache_args = (args.laser, gts)
    dfs = None
    if args.only_plot:
        dfs = madplot.load_bagfile_cache(cache_args, cache_fname)
    if dfs is None:
        dfs = prepare_data(path, args.laser, gts)
        madplot.save_bagfile_cache(dfs, cache_args, cache_fname)

    plot_data(path, args.laser, dfs)

    #p_values1, p_values2 = run_stats(path, dfs)
    #fit_to_curve( p_values1 )
    #fit_to_curve( p_values2 )

    if args.dose_response:
        COLORS = {'100hpc':flymad_plot.BLACK,
                  '120hpc':flymad_plot.GREEN,
                  '140hpc':flymad_plot.RED,
                  '160hpc':flymad_plot.BLUE}

        laser_court = {}
        laser_dtarget = {}
        for laser in [100,120,140,160]:
            laser = '%dhpc' % laser

            cache_fname = os.path.join(path,'courtship_dr_%s.madplot-cache' % laser)
            cache_args = None
            dfs = None

            if args.only_plot:
                dfs = madplot.load_bagfile_cache(cache_args, cache_fname)
            if dfs is None:
                dfs = prepare_data(path, laser, [EXP_GENOTYPE])
                madplot.save_bagfile_cache(dfs, cache_args, cache_fname)

            expdf = dfs[EXP_GENOTYPE]
            laser_court[laser] = dict(xaxis=expdf['mean']['t'].values,
                                      value=expdf['mean']['zx'].values,
                                      std=expdf['std']['zx'].values,
                                      n=expdf['n']['zx'].values,
                                      color=COLORS[laser],
                                      N=len(expdf['df']['obj_id'].unique()))
            laser_dtarget[laser] = dict(xaxis=expdf['mean']['t'].values,
                                        value=expdf['mean']['dtarget'].values,
                                        std=expdf['std']['dtarget'].values,
                                        n=expdf['n']['dtarget'].values,
                                        color=COLORS[laser],
                                        N=len(expdf['df']['obj_id'].unique()))

        #all D/R experiments were identical, so take activation times from the
        #last one
        targetbetween = dict(xaxis=expdf['mean']['t'].values,
                             where=expdf['mean']['laser_state'].values>0)

        figw = plt.figure("Courtship Wingext 10min D/R")
        axw = figw.add_subplot(1,1,1)
        flymad_plot.plot_timeseries_with_activation(axw,
                        targetbetween=targetbetween,
                        sem=True,
                        **laser_court
        )
        axw.set_ylabel('Wing Ext. Index')

        figd = plt.figure("Courtship Dtarget 10min D/R")
        axd = figd.add_subplot(1,1,1)
        flymad_plot.plot_timeseries_with_activation(axd,
                        targetbetween=targetbetween,
                        sem=True,
                        **laser_dtarget
        )
        axd.set_ylabel('Distance (mm)')

        for figname,fig,ax in [("DR_following_and_WingExt",figw,axw), ("DR_following_and_dtarget",figd,axd)]:
            ax.set_xlabel('Time (s)')
            ax.set_xlim([-60,120])
            fig.savefig(flymad_plot.get_plotpath(path,"%s.png" % figname), bbox_inches='tight')
            fig.savefig(flymad_plot.get_plotpath(path,"%s.svg" % figname), bbox_inches='tight')

    if args.show:
        plt.show()

