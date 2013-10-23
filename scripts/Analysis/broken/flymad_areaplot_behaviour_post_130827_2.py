
#INPUT: 1. BACKGROUND.PNG (screenshot from mp4) 2.output from mad_score_movies.py (csv files)
#put all relevant files (one .png and all corresponding .csv) into one folder and call with dir.


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
    
    CTRL_GENOTYPE = 'uasstoptrpmyc' #black
    EXP_GENOTYPE = 'wGP' #blue
    EXP_GENOTYPE2 = '40347' #purple
    

    #PROCESS SCORE FILES:
    pooldf = DataFrame()
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
        print "processing:", csvfilefn         
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
        
        #MATCH COLUMN NAMES (OLD VS NEW flymad_score_movie)
        datenum = int(date)
        if datenum >= 20130827:
            df = df.drop('as',axis=1)
            df = df.rename(columns={'tracked_t':'t', 'laser_state':'as'}) #, inplace=False
            df['as'] = df['as'].fillna(value=0)
        else:
            pass           

        df[['t','theta','v','vx','vy','x','y','zx','as','cv']] = df[['t','theta','v','vx','vy','x','y','zx','as','cv']].astype(float)
        
        #ALIGN BY LASER OFF TIME (=t0)
        if (df['as']==1).any():
            lasermask = df[df['as']==1]
            df['t'] = df['t'] - np.max(lasermask['t'].values)#laser off time =0               
        else:  
            print "No laser detected. Default start time = -120."
            df['t'] = df['t'] - np.min(df['t'].values) - 120
            continue  # throw out for now !!!!
            
        #GENERATE SUBSETS OF TIMEPOINTS
        pre_laser = df[df['t'] <= np.min(lasermask['t'].values) - np.max(lasermask['t'].values)]
        post_laser = df[df['t'] >= 30]
        post_laser = post_laser[post_laser['t'] <=190]
        zbehavmask = df[df['zx']==1]
        cbehavmask = df[df['cv']==1]
        z_prelaser = pre_laser[pre_laser['zx'] ==1]
        z_lasermask = lasermask[lasermask['zx'] ==1]
        z_postlaser = post_laser[post_laser['zx'] ==1]        
        
            
        # OPEN IMAGE
        image_file = str(glob.glob(sys.argv[1] + "/*.png"))[2:-2]
        image_file = open(image_file, 'r')
        image = plt.imread(image_file)
    
        #FIG - PLOT 2D POSITION WITH WING EXTENSION
        
        fig3=plt.figure()
        #fig3.set_size_inches(8,12)
        #fig3.subplots_adjust(hspace=0)
        #plot pre-laser
        ax3=fig3.add_subplot(3,1,1)	
        ax3.set_title('Before Activation', size=12)
        ax3.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax3.scatter(pre_laser['x'], pre_laser['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax3.scatter(z_prelaser['x'], z_prelaser['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax3.set_ylim([495,0])
        ax3.set_xlim([215,680])
        ax3.axis('off') # clear x- and y-axes
        #plot during-laser
        ax4=fig3.add_subplot(3,1,2)	
        ax4.set_title('During Activation', size=12)
        ax4.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax4.scatter(lasermask['x'], lasermask['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax4.scatter(z_lasermask['x'], z_lasermask['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax4.set_ylim([495,0])
        ax4.set_xlim([215,680])
        ax4.axis('off') # clear x- and y-axes    
        #plot post-laser
        ax5=fig3.add_subplot(3,1,3)	
        ax5.set_title('After Activation', size=12)
        ax5.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax5.scatter(post_laser['x'], post_laser['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax5.scatter(z_postlaser['x'], z_postlaser['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax5.set_ylim([495,0])
        ax5.set_xlim([215,680])
        ax5.axis('off') # clear x- and y-axes
        #fig3.subplots_adjust(left= 0.01, right = 0.99, top=0.94, bottom = 0.01, wspace = 0.17, hspace = 0.08)
     
        plt.savefig((sys.argv[1] + '/2Dplot.png'), dpi=600, pad_inches=0)
        plt.savefig((sys.argv[1] + '/2Dplot.svg'), dpi=600, pad_inches=0)
        plt.show() 
    
