
#INPUT: 1. BACKGROUND.PNG (screenshot from mp4) 2.output from mad_score_movies.py (csv files)
#put all relevant files (one .png and all corresponding .csv) into one folder and call with dir.


#OUTPUT: /outputs/*.CSV -- ONE FILE PER ; t0=LASEROFF; dtarget=DISTANCE TO NEAREST TARGET

#P1:     wGP-300-23-130814....
#pIP10:  40347-300-03-130830....
#ctrl:   uasstoptrpmyc-300-03-130828...

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

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis

if __name__ == '__main__':
    if len(sys.argv) !=2:
        print 'call with directory. example: "home/user/foo/filedir"'
        exit()	
    
    CTRL_GENOTYPE = 'uasstoptrpmyc' #black
    EXP_GENOTYPE = 'wGP' #blue
    EXP_GENOTYPE2 = '40347' #purple
    

    #PROCESS SCORE FILES:
    for df,metadata in flymad_analysis.courtship_combine_csvs_to_dataframe(sys.argv[1]):
        csvfilefn,experimentID,date,time,genotype,laser,repID = metadata

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
        z_prelaser = pre_laser[pre_laser['zx'] ==1]
        z_lasermask = lasermask[lasermask['zx'] ==1]
        z_postlaser = post_laser[post_laser['zx'] ==1]        
            
            
        # OPEN IMAGE
        #image_file = str(glob.glob(sys.argv[1] + "/*.png"))[2:-2]
        #image_file = open(image_file, 'r')
        #image = plt.imread(image_file)
    
        #FIG - PLOT 2D POSITION WITH WING EXTENSION
        
        fig3=plt.figure(figsize=(4,12))
        #fig3.set_size_inches(8,12)
        #fig3.subplots_adjust(hspace=0)
        #plot pre-laser
        ax3=fig3.add_subplot(3,1,1)	
        ax3.set_title('Before Activation', size=12)
        #ax3.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax3.scatter(pre_laser['x'], pre_laser['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax3.scatter(z_prelaser['x'], z_prelaser['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax3.set_ylim([495,0])
        ax3.set_xlim([215,680])
        ax3.axis('off') # clear x- and y-axes
        #plot during-laser
        ax4=fig3.add_subplot(3,1,2)	
        ax4.set_title('During Activation', size=12)
        #ax4.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax4.scatter(lasermask['x'], lasermask['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax4.scatter(z_lasermask['x'], z_lasermask['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax4.set_ylim([495,0])
        ax4.set_xlim([215,680])
        ax4.axis('off') # clear x- and y-axes    
        #plot post-laser
        ax5=fig3.add_subplot(3,1,3)	
        ax5.set_title('After Activation', size=12)
        #ax5.imshow(image, extent=[0,1350,495,0],zorder=0) #extent=[h_min,h_max,v_min,v_max]
        ax5.scatter(post_laser['x'], post_laser['y'], zorder=1, marker='o', c='b', s=1, lw=0)
        ax5.scatter(z_postlaser['x'], z_postlaser['y'], zorder=2, marker='o', c='r', s=1, lw=0)
        ax5.set_ylim([495,0])
        ax5.set_xlim([215,680])
        ax5.axis('off') # clear x- and y-axes
        #fig3.subplots_adjust(left= 0.01, right = 0.99, top=0.94, bottom = 0.01, wspace = 0.17, hspace = 0.08)

        plot = os.path.join(sys.argv[1],'outputs',csvfilefn + '_2D.png')
        fig3.savefig(plot)

    
