#!/usr/bin/env python

import roslib
roslib.load_manifest('flymad')

import rosbag

import pandas as pd
import numpy as np
import datetime

def _check_t(msg_t, rt, msg_name=''):
    dt = datetime.datetime.fromtimestamp(msg_t)
    if dt.year < 2013:
        print "WARNING: INVALID TIME IN MSG %s - USING RECEIVED TIME INSTEAD" % msg_name
        dt = datetime.datetime.fromtimestamp(rt)
        if dt.year < 2013:
            raise Exception("INVALID RECEIVED TIME. CHECK YOUR CLOCK")
        return rt
    return msg_t

def pp_df_raw2d(df):
    dt = np.gradient(df['tracked_t'].values)
    df['vx'] = np.gradient(df['x'].values) / dt
    df['vy'] = np.gradient(df['y'].values) / dt
    df['v'] = np.sqrt( (df['vx'].values**2) + (df['vy'].values**2) )

def fmt_msg_raw2d(msg, rt, data_dict):
    if len(msg.points) == 1:
        pt = msg.points[0]
        t = msg.header.stamp.to_sec()

        data_dict["x"].append(pt.x)
        data_dict["y"].append(pt.y)
        data_dict["theta"].append(pt.theta)
        data_dict["tracked_t"].append(t)

        data_dict["t"].append(
                    _check_t(t,rt.to_sec(),'/flymad/raw_2d_positions')
        )

def fmt_msg_generic(msg, rt, data_dict):
    t = rt.to_sec()
    data_dict["data"].append(msg.data)
    data_dict["t"].append(t)

SECOND_TO_NANOSEC = 1e9

TOPICS = [
    #the last component is the name of the member in the msg
    "/flymad/raw_2d_positions/x",
    "/flymad/raw_2d_positions/y",
    "/flymad/raw_2d_positions/theta",
    "/flymad/raw_2d_positions/tracked_t",
    "/flymad_micro/laser/data",
    "/flymad_micro/adc/data",
]

TOPIC_COL_MAP = {
    "/flymad_micro/laser/data" : "laser_state",
    "/flymad_micro/adc/data" : "laser_power",
}

TOPIC_FMT_MAP = {
    "/flymad/raw_2d_positions" : fmt_msg_raw2d,
    "/flymad_micro/laser" : fmt_msg_generic,
    "/flymad_micro/adc" : fmt_msg_generic
}

TOPIC_FMT_MAP = {
    "/flymad/raw_2d_positions" : fmt_msg_raw2d,
    "/flymad_micro/laser" : fmt_msg_generic,
    "/flymad_micro/adc" : fmt_msg_generic
}

TOPIC_POSTPROCESS_MAP = {
    "/flymad/raw_2d_positions" : pp_df_raw2d
}

def create_df(bname):

    #map member name to topics
    topics = {}
    for _t in TOPICS:
        _,n,p,d = _t.split("/")
        t = "/%s/%s" % (n,p)
        try:
            topics[t][d] = []
        except KeyError:
            #reserve space for the time
            topics[t] = {d:[],"t":[]}

    data = {i:[] for i in topics}
    with rosbag.Bag(bname,'r') as b:
        for topic,msg,rt in b.read_messages(topics=topics.keys()):
            func = TOPIC_FMT_MAP[topic]
            data = topics[topic]
            func(msg,rt,data)

    #make one dataframe per topic
    dfs = []
    for t in topics:
        #adjust the names of the colums if needed
        dfdata = {}
        for col in topics[t]:
            if col == "t":
                continue
            fullname = "%s/%s" % (t,col)
            dfdata[ TOPIC_COL_MAP.get(fullname, col) ] = topics[t][col]

        times = topics[t]["t"]
        if len(times) > 0:
            df = pd.DataFrame(
                        dfdata,
                        index=(np.array(times)*SECOND_TO_NANOSEC).astype(np.int64),
            )

            if not df.index.is_unique:
                print "DUPLICATE INDICES (TIMES) IN DATAFRAME"

            try:
                TOPIC_POSTPROCESS_MAP[t](df)
            except KeyError:
                pass

            dfs.append(df)

        else:
            print "NO MESSAGES RECEIVED FOR TOPIC",t

    df = pd.concat(dfs,axis=1)

    return df

if __name__ == "__main__":
    import sys
    fname = sys.argv[1]
    df = create_df(sys.argv[1])
    df.fillna(method='pad').to_csv(fname+".csv")



