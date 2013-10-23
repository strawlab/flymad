
#INPUT: 1. BACKGROUND.PNG (screenshot from mp4) 2.output from mad_score_movies.py (csv files)
###put all relevant files (one .png and all corresponding .csv) into one folder and call with dir.


#OUTPUT: /outputs/*.CSV -- ONE FILE PER ; t0=LASEROFF; dtarget=DISTANCE TO NEAREST TARGET

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

if __name__ == '__main__':
    if len(sys.argv) !=2:
        print 'call with directory. example: "home/user/foo/filedir"'
        exit()	
    
    CTRL_GENOTYPE = 'uasstoptrpmyc'
    EXP_GENOTYPE = '40347'  
    
  
    # OPEN IMAGE of background  (.png)
    image_file = str(glob.glob(sys.argv[1] + "/*.png"))[2:-2]
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
    if not os.path.exists(sys.argv[1] + "/outputs/"):
        os.makedirs(sys.argv[1] + "/outputs/")
    targets.to_csv(sys.argv[1] + '/outputs/targetlocations.csv')

    #PROCESS SCORE FILES:
    pooldf = DataFrame()
    df2 = DataFrame()
    filelist = []
    for posfile in sorted(glob.glob(sys.argv[1] + "/*.csv")):
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
        print "processing:", csvfilefn, genotype         
        for csvfile2 in sorted(glob.glob(sys.argv[1] + "/*.csv")):
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
        
        #MATCH COLUMN NAMES (OLD VS NEW mad_score_movie)
        datenum = int(date)
        if datenum >= 20130827:
            df = df.drop('as',axis=1)
            df = df.rename(columns={'tracked_t':'t', 'laser_state':'as'}) #, inplace=False
            df['as'] = df['as'].fillna(value=0)
        else:
            pass           
    
        #MAKE POSITIONS INTO INTEGERS      
        #df['x'] = (df['x'] + 0.5).astype(int)
        #df['y'] = (df['y'] + 0.5).astype(int) 
        df[['t','theta','v','vx','vy','x','y','zx','as','cv']] = df[['t','theta','v','vx','vy','x','y','zx','as','cv']].astype(float)
          
        #df['x'][np.isfinite(df['x'])] = (df['x'][np.isfinite(df['x'])] + 0.5).astype(int)
        #df['y'][np.isfinite(df['y'])] = (df['y'][np.isfinite(df['y'])] + 0.5).astype(int)
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
            continue  # throw out for now !!!!!!!!!!!!!!!!!!!!!11\
        
        #bin to  1 ms bins:
        df = df[np.isfinite(df['t'])]
        df['t'] = df['t'] *100
        df['t'] = df['t'].astype(int)
        df['t'] = df['t'].astype(float)
        df['t'] = df['t'] /100
        df = df.groupby(df['t'], axis=0).mean() 
        df['r'] = range(0,len(df))
        df['as'][df['as'] > 0] = 1
        df = df.set_index('r') 
        df['Genotype'] = genotype
        df['Genotype'][df['Genotype'] =='csGP'] = 'wGP'
        df['lasergroup'] = laser
        df['RepID'] = repID 
        
        #combine 7s trials together into df2:
        dfshift = df.shift()
        laserons = df[(df['as']-dfshift['as']) == 1]
        prev = -3000 
        for row in laserons.index:
            if (row-prev) <= 5500:
                continue
            else:
                print "row:", row, "\t span:", row - prev
                dftemp = df.ix[(row-990):(row+5800)][['Genotype', 'lasergroup','zx', 'dtarget', 'as']]
                dftemp['align'] = np.arange(-9.90,58.01,0.01)
                #bin to  0.5 second bins:
                dftemp['align'] = dftemp['align'] /0.5
                dftemp['align'] = dftemp['align'].astype(int)
                dftemp['align'] = dftemp['align'].astype(float)
                dftemp['align'] = dftemp['align'] *0.5
                dftemp = dftemp.groupby(dftemp['align'], axis=0).mean()
                dftemp['as'][dftemp['as'] > 0] = 1
                dftemp['Genotype'] = genotype
                dftemp['Genotype'][dftemp['Genotype'] =='csGP'] = 'wGP'
                dftemp['lasergroup'] = laser 
                prev = row #(compare each laseron to the previous to remove wrong laserons.)
                
                df2 = pd.concat([df2, dftemp])

    expdf = df2[df2['Genotype'] == EXP_GENOTYPE]
    ctrldf = df2[df2['Genotype']== CTRL_GENOTYPE]
    expmean = expdf.groupby(['Genotype', 'lasergroup', 'align'], as_index=False)[['zx', 'dtarget', 'as']].mean()
    ctrlmean = ctrldf.groupby(['Genotype', 'lasergroup', 'align'], as_index=False)[['zx', 'dtarget', 'as']].mean()
    expstd = expdf.groupby(['Genotype', 'lasergroup', 'align'], as_index=False)[['zx', 'dtarget', 'as']].std()
    ctrlstd = ctrldf.groupby(['Genotype', 'lasergroup', 'align'], as_index=False)[['zx', 'dtarget', 'as']].std()
    
    expn = expdf.groupby(['Genotype', 'lasergroup','align'],  as_index=False)[['zx', 'dtarget', 'as']].count()
    ctrln = ctrldf.groupby(['Genotype', 'lasergroup','align'],  as_index=False)[['zx', 'dtarget', 'as']].count()
    
    expn.to_csv((sys.argv[1] + "/outputs/expn.csv"))
    expzxsem = ((expstd['zx']).values) / (np.sqrt(expn['zx'].values))
    expdtargetsem = (expstd['dtarget'].values)/(np.sqrt(expn['dtarget'].values))
    
    ctrlzxsem = ((ctrlstd['zx']).values) / (np.sqrt(ctrln['zx'].values))
    ctrldtargetsem = (ctrlstd['dtarget'].values)/(np.sqrt(ctrln['dtarget'].values)) 
    
    expmean.to_csv((sys.argv[1] + "/outputs/expmeans.csv"))
    ctrlmean.to_csv((sys.argv[1] + "/outputs/ctrlmeans.csv"))



    # PLOT FULL TRACE (MEAN +- STD)
    

    fig = plt.figure()
    #WING EXTENSION:
    ax = fig.add_subplot(2,1,1)
    ax.plot(expmean['align'], expmean['zx'], 'b-')
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between((expmean['align']).values, (expmean['zx'] + expzxsem).values, (expmean['zx'] - expzxsem).values, color='b', alpha=0.1, zorder=2)
    ax.plot(ctrlmean['align'], ctrlmean['zx'], 'r-')
    ax.fill_between((ctrlmean['align']).values, (ctrlmean['zx'] + ctrlzxsem).values, (ctrlmean['zx'] - ctrlzxsem).values, color='r', alpha=0.1, zorder=2)
    ax.fill_between((ctrlmean['align']).values, 0, 1, where=(ctrlmean['as']>0.9).values, facecolor='Yellow', alpha=0.3, transform=trans)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wing Ext. Index, +/- SEM')
    ax.set_title('Wing Extension', size=12)
    ax.set_ylim([0,0.5])
    #ax.set_xlim([-60,220])
    #ax.set_xticks([0,60,120,180])
    plt.axhline(y=0, color='k')
  
      
    #DISTANCE TO TARGETS:
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(expmean['align'], expmean['dtarget'], 'b-')
    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.fill_between((expmean['align']).values, (expmean['dtarget'] + expdtargetsem).values, (expmean['dtarget'] - expdtargetsem).values, color='b', alpha=0.1, zorder=2)
    ax2.plot(ctrlmean['align'], ctrlmean['dtarget'], 'r-')
    ax2.fill_between((ctrlmean['align']).values, (ctrlmean['dtarget'] + ctrldtargetsem).values, (ctrlmean['dtarget'] - ctrldtargetsem).values, color='r', alpha=0.1, zorder=2)
    ax2.fill_between((ctrlmean['align']).values, 0, 1, where=(ctrlmean['as']>0.9).values, facecolor='Yellow', alpha=0.3, transform=trans)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (pixels), +/- SEM')
    ax2.set_title('Distance to Nearest Target', size=12)
    #ax2.set_ylim([0,100])
    #ax2.set_xlim([-60,220])
    #ax2.set_xticks([0,60,120,180])

    
    plt.subplots_adjust(bottom=0.1, top=0.94, hspace=0.38)
    plt.savefig((sys.argv[1] + "/outputs/following_and_WingExt.png"))
    plt.savefig((sys.argv[1] + "/outputs/following_and_WingExt.svg"))
    plt.show()


    
