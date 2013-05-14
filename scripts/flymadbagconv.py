import roslib
roslib.load_manifest('flymad')

import rosbag

import pandas as pd

def create_df(bname):
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
                y.append(pt.x)
                theta.append(pt.theta)
                times.append(t)

    df = pd.DataFrame(
            {"x":x,
             "y":y,
             "theta":theta},
            index=times
    )
    return df

if __name__ == "__main__":
    import sys
    print create_df(sys.argv[1])
