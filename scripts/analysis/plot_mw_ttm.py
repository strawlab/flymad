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

def load_bagfile_get_laseron(arena, score):
    FWD = 's'
    BWD = 'a'
    AS_MAP = {FWD:1,BWD:-1}

    l_df, t_df, h_df, geom = madplot.load_bagfile(score.bag, arena)

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

def prepare_data(arena, path, genotype):
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
                t_df,lon = load_bagfile_get_laseron(arena, score)

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

    cPickle.dump(data, open(os.path.join(path,'data_%s_%s.pkl' % (genotype,arena.unit)),'wb'), -1)

    return data

def load_data(arena, path, gt):
    return cPickle.load(open(os.path.join(path,'data_%s_%s.pkl' % (gt,arena.unit)),'rb'))

def plot_data(arena, path, data, genotype):

    targets = data[genotype]['targets']

    for trg in targets:

        print "-- MW %s N --------------\n\t%s = %s" % (genotype,trg,len(targets[trg]))

        pooled_vfwd = {}
        pooled_theta = {}

        vals = []
        figv = plt.figure('velocity %s %s' % (genotype, trg))
        axv = figv.add_subplot(1,1,1)
        figt = plt.figure('dtheta %s %s' % (genotype, trg))
        axt = figt.add_subplot(1,1,1)
        for i,(score,t_df,lon) in enumerate(targets[trg]):

            #forward velocity
            vfwd = t_df['Vfwd']
            ser = pd.Series(vfwd.values, index=np.arange(0,len(vfwd))-lon)
            pser = ser.loc[-100:900]
            axv.plot(pser.index, pser.values,'k',label=os.path.basename(score.mp4),alpha=0.2)

            if len(ser) > 9000:
                pooled_vfwd[i] = ser

            #dtheta
            theta = t_df['theta']
            if not np.any(theta.isnull()):
                dtheta = np.abs(np.gradient(theta.values))

                #discontinuities are wrap arounds, ignore....
                dtheta[dtheta > (0.9*np.pi / 2.0)] = np.nan

                ser = pd.Series(dtheta, index=np.arange(0,len(vfwd))-lon)
                pser = ser.loc[-100:900]
                axt.plot(pser.index, pser.values,'k.',label=os.path.basename(score.mp4),alpha=0.2)

                if len(ser) > 9000:
                    pooled_theta[i] = ser

        m = pd.DataFrame(pooled_vfwd).mean(axis=1)
        pm = m.loc[-100:900]

        axv.plot(pm.index, pm.values,'r',label=os.path.basename(score.mp4),lw=2, alpha=0.8)
        axv.set_xlim([-100,900])
        axv.set_ylabel('velocity (%s/s)' % arena.unit)

        m = pd.DataFrame(pooled_theta).mean(axis=1)
        pm = m.loc[-100:900]
        axt.plot(pm.index, pm.values,'r',label=os.path.basename(score.mp4),lw=2, alpha=0.8)
        axt.set_xlim([-100,900])
        axt.set_ylabel('dtheta (rad/s)')
        axt.set_ylim([0,0.6])

        figv.savefig(os.path.join(path,'velocity_%s_%s.png' % (genotype,trg)))
        figt.savefig(os.path.join(path,'dtheta_%s_%s.png' % (genotype,trg)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='path to mp4s')
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--genotype', default='Moonw')

    args = parser.parse_args()
    path = args.path[0]

    arena = madplot.Arena('mm')

    if args.only_plot:
        data = load_data(arena, path, args.genotype)
    else:
        data = prepare_data(arena, path, args.genotype)

    plot_data(arena, path, data, args.genotype)

    if args.show:
        plt.show()

