import os

opj = os.path.join

_DOROTHEA_BASEDIR = '/mnt/strawscience/data/FlyMAD/revision_dorothea/TH_Gal4_experiments'

DOROTHEA_NAME_RE_BASE = r'(?P<condition>.*)_(?P<condition_flynum>\d+)(_(?P<trialnum>\d))?_(?P<datetime>\d\d\d\d\d\d\d\d_\d\d\d\d\d\d).fmf'

DOROTHEA_BAGDIR = opj(_DOROTHEA_BASEDIR,'bags')
DOROTHEA_FMFDIR = opj(_DOROTHEA_BASEDIR,'fmfs')
DOROTHEA_MP4DIR = opj(_DOROTHEA_BASEDIR,'mp4s')
