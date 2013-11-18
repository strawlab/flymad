import sys
import glob
import os.path
import datetime

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    power = [150,200,250,300]
    protocols = ['shi','MW']

    for pwr in power:
        for proto in protocols:
            name = 'temptest-%d%s-01.dat' % (pwr,proto)
            path = os.path.join('/mnt/strawscience/data/FlyMAD/dans_data/pulse_temp_data/temperature_data_130912', name)

            if not os.path.exists(path):
                continue


            plt.figure(name)
            df = pd.read_csv(path, sep='\t', header=None, skiprows=1,
                             names=['idx','voltage','temperature'],
                             index_col=0,parse_dates=True,
                             date_parser=datetime.datetime.fromtimestamp)

            df['temperature'].plot()

    plt.show()
