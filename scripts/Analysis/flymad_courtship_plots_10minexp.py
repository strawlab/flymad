
#INPUT: 1. BACKGROUND.PNG (screenshot from mp4) 2.output from mad_score_movies.py (csv files)
#put all relevant files (one .png and all corresponding .csv) into one folder and call with dir.


#OUTPUT: /outputs/*.CSV -- ONE FILE PER ; t0=LASEROFF; dtarget=DISTANCE TO NEAREST TARGET

import argparse
import math
import glob
import os
import sys
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import flymad_analysis
import flymad_plot

#need to support numpy datetime64 types for resampling in pandas
assert np.version.version in ("1.7.1", "1.6.1")
assert pd.__version__ == "0.11.0"

def prepare_data(path, exp_genotype, exp2_genotype, ctrl_genotype):
    path_out = path + "/outputs/"


    # OPEN IMAGE of background  (.png)
    image_file = str(glob.glob(path + "/*.png"))[2:-2]
    image_file = open(image_file, 'r')
    image = plt.imread(image_file)
    fig1=plt.figure()
    fig1.set_size_inches(12,8)
    fig1.subplots_adjust(hspace=0)
    ax1=fig1.add_subplot(1,1,1)	
    ax1.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
    ax1.axis('off') 
    # DEFINE TARGET POSITIONS AND SAVE THEM TO A DATAFRAME (targets)    
    targets = []
    def onclick(target):
        print [target.xdata, target.ydata] 
        xydict = {'x': target.xdata, 'y': target.ydata}
        targets.append(xydict)
        return
    cid = fig1.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig1.canvas.mpl_disconnect(cid)
    targets = DataFrame(targets)
    targets = (targets + 0.5).astype(int)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    targets.to_csv(path_out+'/targetlocations.csv')

    #PROCESS SCORE FILES:
    pooldf = DataFrame()
    filelist = []
    for posfile in sorted(glob.glob(path + "/*.csv")):
        csvfilefn = os.path.basename(posfile)
        df = pd.read_csv(posfile)
        try:
            experimentID,date,time = csvfilefn.split("_",2)
            genotype,laser,repID = experimentID.split("-",2)
            repID = experimentID + "_" + date
        except:
            print "invalid filename:", csvfilefn
            continue 
        #CONCATENATE MATCHING IDs:
        if csvfilefn in filelist:
            continue   #avoid processing flies more than once.
        filelist.append(csvfilefn)  
        print "processing:", csvfilefn         
        for csvfile2 in sorted(glob.glob(path + "/*.csv")):
            csvfile2fn = os.path.basename(csvfile2)
            try:
                experimentID2,date2,time2 = csvfile2fn.split("_",2)
                genotype2,laser2,repID2 = experimentID2.split("-",2)
                repID2 = experimentID2 + "_" + date2
            except:
                continue
            if csvfile2fn in filelist:
                continue 
            elif repID2 == repID:
                print "    concatenating:", csvfile2fn
                filelist.append(csvfile2fn)
                csv2df = pd.read_csv(csvfile2)
                csv2df = DataFrame(csv2df)
                df = pd.concat([df, csv2df])
            else:
                continue
  
        #convert 'V', 'X' AND 'S' to 1 or 0
        df['zx'] = df['zx'].astype(object).fillna('x')
        df['as'] = df['as'].astype(object).fillna('s')
        df['cv'] = df['cv'].astype(object).fillna('v')
        df['as'].fillna(value='s')
        df['cv'].fillna(value='v')        
        df['zx'][df['zx'] == 'z'] = 1
        df['cv'][df['cv'] == 'c'] = 1
        df['as'][df['as'] == 'a'] = 1
        df['zx'][df['zx'] == 'x'] = 0
        df['cv'][df['cv'] == 'v'] = 0
        df['as'][df['as'] == 's'] = 0
        
        #MATCH COLUMN NAMES (OLD VS NEW flymad_score_movie)
        datenum = int(date)
        if datenum >= 20130827:
            df = df.drop('as',axis=1)
            df = df.rename(columns={'tracked_t':'t', 'laser_state':'as'}) #, inplace=False
            df['as'] = df['as'].fillna(value=0)
        else:
            pass           

        df[['t','theta','v','vx','vy','x','y','zx','as','cv']] = df[['t','theta','v','vx','vy','x','y','zx','as','cv']].astype(float)

        #CALCULATE DISTANCE FROM TARGETs, KEEP MINIMUM AS dtarget
        dist = DataFrame.copy(df, deep=True)
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
        df = df.sort(columns='t', ascending=True, axis=0)
        
        #ALIGN BY LASER OFF TIME (=t0)
        if (df['as']==1).any():
            lasermask = df[df['as']==1]
            df['t'] = df['t'] - np.max(lasermask['t'].values)#laser off time =0               
        else:  
            print "No laser detected. Default start time = -120."
            df['t'] = df['t'] - np.min(df['t'].values) - 120
            continue  # throw out for now !!!!
             
        #bin to  5 second bins:
        df = df[np.isfinite(df['t'])]
        df['t'] = df['t'] /5
        df['t'] = df['t'].astype(int)
        df['t'] = df['t'].astype(float)
        df['t'] = df['t'] *5
        df = df.groupby(df['t'], axis=0).mean() 
        df['r'] = range(0,len(df))
        df['as'][df['as'] > 0] = 1
        df = df.set_index('r') 
        df['Genotype'] = genotype
        df['Genotype'][df['Genotype'] =='csGP'] = 'wGP'
        df['lasergroup'] = laser
        df['RepID'] = repID             

        df.to_csv((path_out + experimentID + "_" + date + ".csv"), index=False)
        pooldf = pd.concat([pooldf, df[['Genotype','lasergroup', 't','zx', 'dtarget', 'as']]])   

    # half-assed, uninformed danno's tired method of grouping for plots:
    expdf = pooldf[pooldf['Genotype'] == exp_genotype]
    exp2df = pooldf[pooldf['Genotype'] == exp2_genotype]
    ctrldf = pooldf[pooldf['Genotype']== ctrl_genotype]

    #we no longer need to group by genotype, and lasergroup is always the same here
    #so just drop it. 
    assert len(expdf['lasergroup'].unique()) == 1, "only one lasergroup handled"

    #Also ensure things are floats before plotting can fail, which it does because
    #groupby does not retain types on grouped colums, which seems like a bug to me

    expmean = expdf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].mean().astype(float)
    exp2mean = exp2df.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].mean().astype(float)
    ctrlmean = ctrldf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].mean().astype(float)
    
    expstd = expdf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].std().astype(float)
    exp2std = exp2df.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].std().astype(float)
    ctrlstd = ctrldf.groupby(['t'], as_index=False)[['zx', 'dtarget', 'as']].std().astype(float)
    
    expn = expdf.groupby(['t'],  as_index=False)[['zx', 'dtarget', 'as']].count().astype(float)
    exp2n = exp2df.groupby(['t'],  as_index=False)[['zx', 'dtarget', 'as']].count().astype(float)
    ctrln = ctrldf.groupby(['t'],  as_index=False)[['zx', 'dtarget', 'as']].count().astype(float)
    
    ####AAAAAAAARRRRRRRRRRRGGGGGGGGGGGGGGHHHHHHHHHH so much copy paste here
    expdf.save(path+'/exp.df')
    exp2df.save(path+'/exp2.df')
    ctrldf.save(path+'/ctrl.df')
    
    expmean.save(path+'/expmean.df')
    exp2mean.save(path+'/exp2mean.df')
    ctrlmean.save(path+'/ctrlmean.df')

    expstd.save(path+'/expstd.df')
    exp2std.save(path+'/exp2std.df')
    ctrlstd.save(path+'/ctrlstd.df')

    expn.save(path+'/expn.df')
    exp2n.save(path+'/exp2n.df')
    ctrln.save(path+'/ctrln.df')

    return (expdf, exp2df, ctrldf, expmean, exp2mean, ctrlmean, expstd, exp2std, ctrlstd, expn, exp2n, ctrln)

def load_data(path):

    return (
        pd.load(path+'/exp.df'),
        pd.load(path+'/exp2.df'),
        pd.load(path+'/ctrl.df'),
        pd.load(path+'/expmean.df'),
        pd.load(path+'/exp2mean.df'),
        pd.load(path+'/ctrlmean.df'),
        pd.load(path+'/expstd.df'),
        pd.load(path+'/exp2std.df'),
        pd.load(path+'/ctrlstd.df'),
        pd.load(path+'/expn.df'),
        pd.load(path+'/exp2n.df'),
        pd.load(path+'/ctrln.df')
    )


def plot_data(path, expdf, exp2df, ctrldf, expmean, exp2mean, ctrlmean, expstd, exp2std, ctrlstd, expn, exp2n, ctrln):
    path_out = path + "/outputs/"

    expzxsem = ((expstd['zx']).values) / (np.sqrt(expn['zx'].values))
    expdtargetsem = (expstd['dtarget'].values)/(np.sqrt(expn['dtarget'].values))

    exp2zxsem = ((exp2std['zx']).values) / (np.sqrt(exp2n['zx'].values))
    exp2dtargetsem = (exp2std['dtarget'].values)/(np.sqrt(exp2n['dtarget'].values))
    
    ctrlzxsem = ((ctrlstd['zx']).values) / (np.sqrt(ctrln['zx'].values))
    ctrldtargetsem = (ctrlstd['dtarget'].values)/(np.sqrt(ctrln['dtarget'].values)) 
    
    fig = plt.figure()
    #WING EXTENSION:
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(expmean['t'].values, expmean['zx'].values, 'b-', zorder=1, lw=3)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(expmean['t'].values, (expmean['zx'] + expzxsem).values, (expmean['zx'] - expzxsem).values, color='b', alpha=0.1, zorder=4)
    plt.axhline(y=0, color='k')
    
    ax.fill_between(expmean['t'].values, 0, 1, where=expmean['as'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=1)
    
    ax.plot(ctrlmean['t'].values, ctrlmean['zx'].values, 'k-', zorder=1, lw=3)
    ax.fill_between(ctrlmean['t'].values, (ctrlmean['zx'] + ctrlzxsem).values, (ctrlmean['zx'] - ctrlzxsem).values, color='k', alpha=0.1, zorder=3)

    #ax.fill_between(ctrlmean['t'].values, 0, 1, where=ctrlmean['as'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=100)

    ax.plot(exp2mean['t'].values, exp2mean['zx'].values, color='purple', zorder=1, lw=3)
    ax.fill_between(exp2mean['t'].values, (exp2mean['zx'] + exp2zxsem).values, (exp2mean['zx'] - exp2zxsem).values, color='purple', alpha=0.1, zorder=2)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wing Ext. Index, +/- SEM')
    ax.set_title('Wing Extension', size=12)
    ax.set_ylim([-0.1,0.6])
    ax.set_xlim([-120,480])
    ax.set_xticks([-60,0,60,120,180,240,300,360,420,480])

    """  
      
    #DISTANCE TO TARGETS:
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(expmean['t'].values, expmean['dtarget'].values, 'b-', zorder=1, lw=3, label = 'P1-TRP')
    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.fill_between(expmean['t'].values, (expmean['dtarget'] + expdtargetsem).values, (expmean['dtarget'] - expdtargetsem).values, color='b', alpha=0.1, zorder=2)
    ax2.fill_between(expmean['t'].values, 0, 1, where=expmean['as'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=100)
    ax2.plot(ctrlmean['t'].values, ctrlmean['dtarget'].values, 'r-', zorder=1, lw=3, label = 'Control')
    ax2.fill_between(ctrlmean['t'].values, (ctrlmean['dtarget'] + ctrldtargetsem).values, (ctrlmean['dtarget'] - ctrldtargetsem).values, color='r', alpha=0.1, zorder=3)
    ax2.fill_between(ctrlmean['t'].values, 0, 1, where=ctrlmean['as'].values>0, facecolor='Yellow', alpha=0.15, transform=trans, zorder=100)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (pixels), +/- SEM')
    ax2.set_title('Distance to Nearest Target', size=12)
    ax2.set_ylim([20,120])
    ax2.set_xlim([-120,480])
    ax2.set_xticks([-60,0,60,120,180,240,300,360,420,480])
    """
    
    plt.subplots_adjust(bottom=0.1, top=0.94, hspace=0.38)
    plt.savefig((path_out + "following_and_WingExt.png"))
    plt.savefig((path_out + "following_and_WingExt.svg"))

if __name__ == "__main__":
    CTRL_GENOTYPE = 'uasstoptrpmyc' #black
    EXP_GENOTYPE = 'wGP' #blue
    EXP_GENOTYPE2 = '40347' #purple

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--only-plot', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path)
    else:
        data = prepare_data(path, EXP_GENOTYPE, EXP_GENOTYPE2, CTRL_GENOTYPE)

    plot_data(path, *data)

    plt.show()
