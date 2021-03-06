import re
import os.path
import time
import glob
import collections

BAG_DATE_FMT = "%Y-%m-%d-%H-%M-%S"
BAG_FILENAME_REGEX = "(?:rosbagOut_)?(?P<desc>[\w\-]+?)?_?(?P<date>(201[0-9][0-9\-]{15})|(201[0-9][0-9_]{11}))\.bag"
MP4_DATE_FMT = "%Y%m%d_%H%M%S"
MP4_FILENAME_REGEX = "(?P<desc>[\-\w]+?)_?(?P<date>201[0-9]+[0-9]{4}_[0-9]{6})\..*"

FMF_DATE_FMT = MP4_DATE_FMT
FMF_FILENAME_REGEX = "(?P<desc>[\w\-]+?)_?(?P<camn>wide|zoom)?_(?P<date>201[0-9]+[0-9]{4}_[0-9]{6})\..*"

_MATCH_PATTERNS = {
    "bag":(BAG_DATE_FMT,BAG_FILENAME_REGEX),
    "mp4":(MP4_DATE_FMT,MP4_FILENAME_REGEX),
    "fmf":(FMF_DATE_FMT,FMF_FILENAME_REGEX),
    "csv":(MP4_DATE_FMT,MP4_FILENAME_REGEX),
}

class RegexError(Exception):
    pass

def parse_filename(path, regex=None, extract_genotype_and_laser=False):
    if regex is None:
        _,regex = _MATCH_PATTERNS[os.path.splitext(path)[1][1:]]
    matchobj = re.search(regex, os.path.basename(path))
    if matchobj is None:
        raise RegexError("incorrectly named file: %s" % path)

    res = matchobj.groupdict()

    if extract_genotype_and_laser and res.get('desc'):
        desc = res['desc']
        try:
            gt,laser,repid = desc.split('-')
        except ValueError:
            gt,laser = desc.split('-')
            repid = None
        res['genotype'] = gt
        res['laser'] = laser
        res['repid'] = repid

    return res

def parse_date(path, regex=None):
    if regex is None:
        _,regex = _MATCH_PATTERNS[os.path.splitext(path)[1][1:]]
    datestr = parse_filename(path, regex).get('date','')
    for fmt in (BAG_DATE_FMT, MP4_DATE_FMT, FMF_DATE_FMT):
        try:
            return time.strptime(datestr, fmt)
        except ValueError:
            pass
    raise RegexError("incorrect date string: %s" % datestr)

def get_matching_files(dira, exta, dirb, extb, maxdt=20):
    da,ra = _MATCH_PATTERNS[exta]
    db,rb = _MATCH_PATTERNS[extb]
    klass = collections.namedtuple(
                'MatchPair',
                '%(exta)s %(exta)s_info %(exta)s_time '\
                '%(extb)s %(extb)s_info %(extb)s_time '\
                'dt' % {'exta':exta,'extb':extb})

    afiles = glob.glob(os.path.join(dira,'*.%s' % exta))
    bfiles = glob.glob(os.path.join(dirb,'*.%s' % extb))

    matched = []

    for a in afiles:
        try:
            ainfo = parse_filename(a, ra)
            atime = parse_date(a, ra)
        except RegexError, e:
            print "invalid filename", a
            continue

        for b in bfiles:
            try:
                binfo = parse_filename(b,rb)
                btime = parse_date(b,rb)
            except RegexError, e:
                print "invalid filename", b
                continue

            dt = abs(time.mktime(atime) - time.mktime(btime))

            if (dt < maxdt):
                matched.append(klass(a,ainfo,atime,b,binfo,btime,dt))

    return matched

if __name__ == "__main__":
    from pprint import pprint

    files = ("rosbagOut_2014-02-23-16-28-08.bag",
             "wGP-140hpc-01_20140223_162808.mp4",
             "wGP-140hpc-01_20140223_162808.mp4.csv",
             "wGP-140hpc-01_wide_20140223_162809.fmf",
             "db194-ok371-05_20140523_131735.bag",
             "wshits-120t-08_20140307_125321.mp4.csv",
             "OK371shits-nolaser-05_20140227_141405.mp4.csv",
    )

    for f in files:
        print f,
        try:
            pprint( parse_filename(f, extract_genotype_and_laser=True) )
        except RegexError:
            print "regex failed"

