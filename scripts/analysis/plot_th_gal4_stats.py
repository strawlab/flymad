#!/usr/bin/env python
import pandas as pd
import numpy as np
import strawlab_mpl.defaults as smd
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds
import matplotlib

def setup_defaults():
    rcParams = matplotlib.rcParams

    rcParams['legend.numpoints'] = 1
    rcParams['legend.fontsize'] = 'medium' #same as axis
    rcParams['legend.frameon'] = False
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1

smd.setup_defaults()
setup_defaults()

import pandas as pd
import sys, os,re
import glob
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3) # prevent plots from changing

if 1:
    measurements = ['wing','jump','abdomen','proboscis']
    for measurement in measurements:
        fname = '%s.df'%(measurement,)
        df_all = pd.read_pickle(fname)

        fig = plt.figure(measurement,figsize=(1.5,2))
        ax = fig.add_subplot(111)
        tick_labels = []
        xticks = []
        condition_number=-1
        for name_key, group1 in df_all.groupby("name_key",sort=True):
            condition_number+=1
            this_x_value = condition_number
            this_y_values = group1['latency'].values
            this_x_values = np.array([this_x_value]*len(this_y_values))
            uw = 0.2
            this_x_values = np.array(this_x_values) + np.random.uniform(-uw, uw, size=(len(this_y_values),))
            ax.plot( this_x_values, this_y_values, 'k.' )
            xticks.append( this_x_value )
            tick_labels.append( name_key )
        spine_placer(ax, location='left')#,bottom' )
        ax.set_yticks([0,10])
        ax.set_xticks(xticks)
        ax.set_xticklabels(tick_labels, rotation='vertical')
        ax.set_ylabel('latency (sec)')
        fig.subplots_adjust(left=0.3,bottom=0.65) # do not clip text
        fig.savefig('th_gal4_latency_%s.svg'%(measurement,))
