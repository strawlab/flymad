#!/usr/bin/env python

import roslib
roslib.load_manifest('flymad')

import rosbag

import pandas as pd
import numpy as np

SECOND_TO_NANOSEC = 1e9

def create_df(bname, calc_vel=True):
    times = []
    x = []
    y = []
    theta = []

    with rosbag.Bag(bname,'r') as b:
        for topic,msg,rt in b.read_messages(topics='/flymad/raw_2d_positions'):
            t = msg.header.stamp.to_sec()
            if len(msg.points) == 1:
                pt = msg.points[0]
                x.append(pt.x)
                y.append(pt.y)
                theta.append(pt.theta)
                times.append(t)

    df = pd.DataFrame({
             "x":x,
             "y":y,
             "theta":theta,
             "t":times,
            },
            index=(np.array(times)*SECOND_TO_NANOSEC).astype(np.int64),
    )
    if not df.index.is_unique:
        print "DUPLICATE INDICES (TIMES) IN DATAFRAME"

    if calc_vel:
        dt = np.gradient(df['t'].values)
        df['vx'] = np.gradient(df['x'].values) / dt
        df['vy'] = np.gradient(df['y'].values) / dt
        df['v'] = np.sqrt( (df['vx'].values**2) + (df['vy'].values**2) )

    return df

if __name__ == "__main__":
    import sys
    print create_df(sys.argv[1])
