import re
import os.path
import collections
import cPickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import roslib; roslib.load_manifest('flymad')
import rosbag

import madplot
import generate_mw_ttm_movies
import flymad.trackingparams

Scored = collections.namedtuple('Scored', 'fmf bag mp4 csv')

def load_bagfile_get_laseron(score, smooth=False):
    FWD = 's'
    BWD = 'a'
    AS_MAP = {FWD:1,BWD:-1}

    arena = madplot.Arena()
    l_df, t_df, h_df, geom = madplot.load_bagfile(score.bag, arena)

    #FIXME: multi smoothing must be implemented better in the load_bagfile
    #function
    #
    if smooth and (len(t_df['tobj_id'].dropna().unique()) == 1):
        #we need dt in seconds to calculate velocity. numpy returns nanoseconds here
        #because this is an array of type datetime64[ns] and I guess it retains the
        #nano part when casting
        dt = np.gradient(t_df.index.values.astype('float64')/1e9)

        #smooth the positions, and recalculate the velocitys based on this.
        kf = flymad.trackingparams.Kalman()
        smoothed = kf.smooth(t_df['x'].values, t_df['y'].values)
        _x = smoothed[:,0]
        _y = smoothed[:,1]
        _vx = np.gradient(_x) / dt
        _vy = np.gradient(_y) / dt
        _v = np.sqrt( (_vx**2) + (_vy**2) )

        t_df['x'] = _x
        t_df['y'] = _y
        t_df['vx'] = _vx
        t_df['vy'] = _vy
        t_df['v'] = _v

    scored_ix = [t_df.index[0]]
    scored_v = [AS_MAP[FWD]]

    for idx,row in pd.read_csv(score.csv).iterrows():
        matching = t_df[t_df['t_framenumber'] == int(row['framenumber'])]
        if len(matching) == 1:
            scored_ix.append(matching.index[0])
            scored_v.append(AS_MAP[row['as']])

    s = pd.Series(scored_v,index=scored_ix)
    t_df['fwd'] = s
    t_df['fwd'].fillna(method='ffill', inplace=True)
    t_df['Vfwd'] = t_df['v'] * t_df['fwd']

    lon = l_df[l_df['laser_power'] > 0]

    if len(lon):
        ton = lon.index[0]

        l_off_df = l_df[ton:]
        loff = l_off_df[l_off_df['laser_power'] == 0]
        toff = loff.index[0]

        onoffset = np.where(t_df.index >= ton)[0][0]
        offoffset = np.where(t_df.index >= toff)[0][0]

        return t_df, onoffset

    else:
        return None, None

def prepare_data(path, genotype):
    GENOTYPES = (genotype,)

    data = {}
    for gt in GENOTYPES:
        bag_re = generate_mw_ttm_movies.get_bag_re(gt)
        targets = {}
        for pair in generate_mw_ttm_movies.get_matching_fmf_and_bag(gt, path):
            mp4dir = os.path.join(os.path.dirname(pair.fmf), 'mp4s')
            mp4 = os.path.join(mp4dir, os.path.basename(pair.fmf))+'.mp4'
            csv = mp4+'.csv'
            if os.path.isfile(csv):
                fmfname = os.path.basename(pair.fmf)
                target,trial,year,date = bag_re.search(fmfname).groups()

                score = Scored(pair.fmf, pair.bag, mp4, csv)
                t_df,lon = load_bagfile_get_laseron(score)

                if t_df is None:
                    print "skip",score.mp4
                    continue

                try:
                    targets[target].append( (score,t_df,lon) )
                except KeyError:
                    targets[target] = [(score,t_df,lon)]
            else:
                print "missing csv for",mp4

        data[gt] = {'targets':targets}

    cPickle.dump(data, open(os.path.join(path,'data_%s.pkl' % genotype),'wb'), -1)

    return data

def load_data(path, gt):
    return cPickle.load(open(os.path.join(path,'data_%s.pkl' % gt),'rb'))

def plot_data(path, data, genotype):
    targets = data[genotype]['targets']

    for trg in targets:

        print "-- MW %s N --------------\n\t%s = %s" % (genotype,trg,len(targets[trg]))

        pooled = {}

        vals = []
        fig = plt.figure(trg)
        ax = fig.add_subplot(1,1,1)
        for i,(score,t_df,lon) in enumerate(targets[trg]):

            vfwd = t_df['Vfwd']
            ser = pd.Series(vfwd.values, index=np.arange(0,len(vfwd))-lon)

            pooled[i] = ser
            ax.plot(ser.index, ser.values,'k',label=os.path.basename(score.mp4),alpha=0.2)

        m = pd.DataFrame(pooled).mean(axis=1)
        ax.plot(m.index, m.values,'r',label=os.path.basename(score.mp4),lw=2, alpha=0.8)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to mp4s')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--genotype', default='Moonw')

    args = parser.parse_args()
    path = args.path[0]

    if args.only_plot:
        data = load_data(path, args.genotype)
    else:
        data = prepare_data(path, args.genotype)

    plot_data(path, data, args.genotype)

    if args.show:
        plt.show()

