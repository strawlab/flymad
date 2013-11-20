import sys
import glob

import matplotlib.pyplot as plt
import numpy as np

import roslib
roslib.load_manifest('flymad')

import rosbag

#multi fly
trials = [1,2,3,4,5,6,7,8]
results = {k:[] for k in trials}
times = {k:[] for k in trials}

for t in trials:

    bf = glob.glob('/mnt/strawscience/data/FlyMAD/temp_mes_dorothea/%df/*.bag' % t)[-1]
    print bf

    with rosbag.Bag(bf) as bag:
        for topic,msg,rostime in bag.read_messages(topics=["/flymad/fly_temperature"]):
            times[t].append(rostime.to_sec())
            results[t].append(msg.data)

figm = plt.figure("Multi Targeter")
ax = figm.add_subplot(1,1,1)
for t in trials:
    startmean = np.array(results[t])[:175].mean()
    time = np.array(times[t]) - times[t][0]
    data = np.array(results[t])
    ax.plot(time, data-startmean, label='%d flies' % t)
ax.legend()

trials = ['2013-11-19-23-35-31.bag','2013-11-19-23-37-43.bag']
results = {k:[] for k in trials}
times = {k:[] for k in trials}

for t in trials:
    bf = '/mnt/strawscience/data/FlyMAD/temp_latency/%s' % t
    print bf

    with rosbag.Bag(bf) as bag:
        for topic,msg,rostime in bag.read_messages(topics=["/flymad/fly_temperature"]):
            times[t].append(rostime.to_sec())
            results[t].append(msg.data)

figt = plt.figure("Temperature Rise")
ax = figt.add_subplot(1,1,1)
for t in trials:
    startmean = 0
    time = np.array(times[t]) - times[t][0]
    data = np.array(results[t])
    ax.plot(time, data-startmean, label=t)
ax.legend()

plt.show()

