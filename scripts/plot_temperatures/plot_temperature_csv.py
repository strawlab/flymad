import glob
import os.path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import roslib; roslib.load_manifest('flymad')
import flymad.flymad_analysis_dan as flymad_analysis
import flymad.flymad_plot as flymad_plot

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to csv files')
    parser.add_argument('--show', action='store_true', default=False)

    args = parser.parse_args()
    path = args.path[0]

    power = [250,300,350]
    area = 'east'

    data = {}

    for pwr in power:
        name = '%d_%s.csv' % (pwr,area)
        path = os.path.join(args.path[0], name)

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, sep=',', header=None, skiprows=2,
                             names=['sample','temperature','time'],
                             index_col=2,parse_dates=True,
                             date_parser=datetime.datetime.fromtimestamp)
        data[pwr] = df.resample('100L')['temperature'].values

    fig = plt.figure()
    ax = fig.gca()
    for pwr in data:
        dat = data[pwr][:890]
        idx = np.array(range(0,len(dat))) * 0.1
        ax.plot(idx, dat, label=str(pwr))
    ax.legend()

    if args.show:
        plt.show()
