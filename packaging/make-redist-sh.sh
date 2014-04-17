#!/bin/bash -x
set -e

#########################################
#
# Generate a .sh file to install FlyMAD.
#
#########################################
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REV=`git rev-parse --short --verify HEAD`

# change into "flymad/packaging/"
cd ${THISDIR}
rm -rf archivedir

mkdir archivedir

cp install-flymad.sh archivedir/
cp install-flymad-prereqs.sh archivedir/
cp ensure_line.py archivedir/
#cp flymad-hydro.rosinstall archivedir/flymad-hydro.rosinstall

# change into "flymad" parent directory for git archive
cd ..
git archive --prefix flymad-${REV}/ --format tar.gz --output ${THISDIR}/archivedir/flymad-${REV}.tar.gz ${REV}
# change back into this directory
cd ${THISDIR}

# Now build the installer script
makeself.sh --base64 --help-header help-header.txt archivedir install-flymad-on-ubuntu-12.04-amd64.sh \
  "FlyMAD installer" ./install-flymad.sh ${REV}
