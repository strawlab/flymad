import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset

import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot
from scipy.stats import kruskal

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.version.version in ("0.11.0" ,  "0.12.0")

def prepare_data(path, exp_genotype, ctrl_genotype):

    df2 = DataFrame()
    if not os.path.exists(path + "/Speed_calculations/"):
        os.makedirs(path + "/Speed_calculations/")
    for csvfile in sorted(glob.glob(path + "/*.csv")):
        csvfilefn = os.path.basename(csvfile)
        try:
            experimentID,date,time = csvfilefn.split("_",2)
            genotype,laser,repID = experimentID.split("-",2)
            repID = repID + "_" + date
            print "processing: ", experimentID
        except:
            print "invalid filename:", csvfilefn
            continue 
        df = pd.read_csv(csvfile, index_col=0)

        if not df.index.is_unique:
            raise Exception("CORRUPT CSV. INDEX (NANOSECONDS SINCE EPOCH) MUST BE UNIQUE")


        #resample to 10ms (mean) and set a proper time index on the df
        df = flymad_analysis.fixup_index_and_resample(df, '10L')

        #smooth the positions, and recalculate the velocitys based on this.
        dt = flymad_analysis.kalman_smooth_dataframe(df)

        df['laser_state'] = df['laser_state'].fillna(value=0)

        lasermask = df[df['laser_state'] == 1]
        df['tracked_t'] = df['tracked_t'] - np.min(lasermask['tracked_t'].values)

        #MAXIMUM SPEED = 300:
        df['v'][df['v'] >= 300] = np.nan
        
        #the resampling above, using the default rule of 'mean' will, if the laser
        #was on any time in that bin, increase the mean > 0.
        df['laser_state'][df['laser_state'] > 0] = 1
            
        df['Genotype'] = genotype
        df['lasergroup'] = laser
        df['RepID'] = repID

        #combine 60s trials together into df2:
        dfshift = df.shift()
        laserons = df[ (df['laser_state']-dfshift['laser_state']) == 1 ]    
        #SILLY BUG. Sometimes laserons contains incorrect timepoints. To fix this, I compare each laseron to the 
        # previous and discard if ix - prev is less than the experimental looping time.
        prev = pd.DatetimeIndex([datetime.datetime(1986, 5, 27)])[0]  #set to random time initially
        for ix,row in laserons.iterrows():
            before = ix - DateOffset(seconds=9.95)
            after = ix + DateOffset(seconds=59.95)
            if (ix - prev).total_seconds() <= 59.95:
                continue
            else:
                print "ix:", ix, "\t span:", ix - prev
                prev = ix
                dftemp = df.ix[before:after][['Genotype', 'lasergroup','v', 'laser_state']]
                dftemp['align'] = np.linspace(0,(after-before).total_seconds(),len(dftemp))
                df2 = pd.concat([df2, dftemp])

    expdf = df2[df2['Genotype'] == exp_genotype]
    ctrldf = df2[df2['Genotype']== ctrl_genotype]

    #we no longer need to group by genotype, and lasergroup is always the same here
    #so just drop it. 
    assert len(expdf['lasergroup'].unique()) == 1, "only one lasergroup handled"

    expmean = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlmean = ctrldf.groupby(['align'], as_index=False).mean().astype(float)

    expstd = expdf.groupby(['align'], as_index=False).mean().astype(float)
    ctrlstd = ctrldf.groupby(['align'], as_index=False).mean().astype(float)

    expn = expdf.groupby(['align'], as_index=False).count().astype(float)
    ctrln = ctrldf.groupby(['align'], as_index=False).count().astype(float)

    ####AAAAAAAARRRRRRRRRRRGGGGGGGGGGGGGGHHHHHHHHHH so much copy paste here
    df2.save(path + "/df2.df.dan")
    expmean.save(path + "/expmean.df.dan")
    ctrlmean.save(path + "/ctrlmean.df.dan")
    expstd.save(path + "/expstd.df.dan")
    ctrlstd.save(path + "/ctrlstd.df.dan")
    expn.save(path + "/expn.df.dan")
    ctrln.save(path + "/ctrln.df.dan")

    return expmean, ctrlmean, expstd, ctrlstd, expn, ctrln, df2

def load_data( path ):
    return (
            pd.load(path + "/expmean.df.dan"),
            pd.load(path + "/ctrlmean.df.dan"),
            pd.load(path + "/expstd.df.dan"),
            pd.load(path + "/ctrlstd.df.dan"),
            pd.load(path + "/expn.df.dan"),
            pd.load(path + "/ctrln.df.dan"),
            pd.load(path + "/df2.df.dan"),
    )

def get_stats(group):
    return {'mean': group.mean(),
            'var' : group.var(),
            'n' : group.count()
           }

def add_obj_id(df):
    results = np.zeros( (len(df),), dtype=np.int )
    obj_id = 0
    for i,(ix,row) in enumerate(df.iterrows()):
        if row['align']==0.0:
            obj_id += 1
        results[i] = obj_id
    df['obj_id']=results
    return df

def calc_kruskal(df_ctrl, df_exp, number_of_bins, align_colname='align', vfwd_colname='v'):
    df_ctrl = add_obj_id(df_ctrl)
    df_exp = add_obj_id(df_exp)

    dalign = df_ctrl['align'].max() - df_ctrl['align'].min()

    p_values = DataFrame()
    for binsize in number_of_bins:
        bins = np.linspace(0,dalign,binsize)
        binned_ctrl = pd.cut(df_ctrl['align'], bins, labels= bins[:-1])
        binned_exp = pd.cut(df_exp['align'], bins, labels= bins[:-1])
        for x in binned_ctrl.levels:
            test1_all_flies_df = df_ctrl[binned_ctrl == x]
            test1 = []
            for obj_id, fly_group in test1_all_flies_df.groupby('obj_id'):
                test1.append( np.mean(fly_group['v'].values) )
            test1 = np.array(test1)

            test2_all_flies_df = df_exp[binned_exp == x]
            test2 = []
            for obj_id, fly_group in test2_all_flies_df.groupby('obj_id'):
                test2.append( np.mean(fly_group['v'].values) )
            test2 = np.array(test2)

            hval, pval = kruskal(test1, test2)
            dftemp = DataFrame({'Total_bins': binsize , 'Bin_number': x, 'P': pval}, index=[x])
            p_values = pd.concat([p_values, dftemp])
    return p_values

def run_stats (path, exp_genotype, ctrl_genotype, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln , df2):
    number_of_bins = [ 6990//4 ]
    df_ctrl = df2[df2['Genotype'] == ctrl_genotype]
    df_exp = df2[df2['Genotype'] == exp_genotype]
    return calc_kruskal(df_ctrl, df_exp, number_of_bins)

def fit_to_curve ( p_values ):
    x = np.array(p_values['Bin_number'][p_values['Bin_number'] <= 50])
    logs = -1*(np.log(p_values['P'][p_values['Bin_number'] <= 50]))
    y = np.array(logs)
    # order = 11 #DEFINE ORDER OF POLYNOMIAL HERE.
    # poly_params = np.polyfit(x,y,order)
    # polynom = np.poly1d(poly_params)
    # xPoly = np.linspace(0, max(x), 100)
    # yPoly = polynom(xPoly)
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ax.plot(x, y, 'bo-')

    ax.axvspan(10,20,
               facecolor='Yellow', alpha=0.15,
               edgecolor='none',
               zorder=-20)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('-log(p)')
    ax.set_ylim([0, 25])
    ax.set_xlim([5, 40])

def plot_data( path, expmean, ctrlmean, expstd, ctrlstd, expn, ctrln, df2):

    fig2 = plt.figure("Speed Multiplot")
    ax = fig2.add_subplot(1,1,1)

    flymad_plot.plot_timeseries_with_activation(ax,
                    exp=dict(xaxis=expmean['align'].values,
                             value=expmean['v'].values,
                             std=expstd['v'].values,
                             n=expn['v'].values,
                             label='OK371>ShibireTS',
                             ontop=True),
                    ctrl=dict(xaxis=ctrlmean['align'].values,
                              value=ctrlmean['v'].values,
                              std=ctrlstd['v'].values,
                              n=ctrln['v'].values,
                              label='Control'),
                    targetbetween=ctrlmean['laser_state'].values>0,
                    downsample=25
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (pixels/s) +/- STD')
    ax.set_ylim([0, 160])
    ax.set_xlim([0, 70])

    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.png"), bbox_inches='tight')
    plt.savefig(flymad_plot.get_plotpath(path,"speed_plot.svg"), bbox_inches='tight')


if __name__ == "__main__":
    EXP_GENOTYPE = 'OK371shib'
    CTRL_GENOTYPE = 'wshib'

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path, EXP_GENOTYPE, CTRL_GENOTYPE)

    p_values = run_stats(path, EXP_GENOTYPE, CTRL_GENOTYPE, *data)
    p_values.to_csv(path + '/p_values.csv')
    fit_to_curve( p_values )
    plot_data(path, *data)

    if args.show:
        plt.show()

