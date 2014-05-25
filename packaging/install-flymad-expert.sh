#!/bin/bash -x
set -e

if [ "$ROS_ROOT" != "" ]; then
    echo "ERROR: cannot run within existing ROS environment"
    exit 1
fi

ROSINSTALL_SPEC_PATH="/tmp/flymad-git.rosinstall"
THISDIR=`pwd`

if [ "$UID" -ne "0" ]; then
    echo "ERROR: you need to have superuser permissions. Re-run with 'sudo'."
    exit 1
fi

# ---- create a .rosinstall spec file for this git revision -------------
cat > ${ROSINSTALL_SPEC_PATH} <<EOF
- git: {local-name: flymad, uri: 'https://github.com/strawlab/flymad.git', version: resubmission-tarball}
- git: {local-name: rosgobject, uri: 'https://github.com/strawlab/rosgobject.git',
    version: flymad-hydro}
EOF

# ---- install our ROS stuff
export FLYMAD_TARGET="/opt/ros/ros-flymad.hydro"
rosinstall ${FLYMAD_TARGET} /opt/ros/hydro/.rosinstall ${ROSINSTALL_SPEC_PATH}

source ${FLYMAD_TARGET}/setup.bash

if [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  echo "rosdep already initialized"
else
  rosdep init
fi

./ensure_line.py "yaml https://raw.github.com/strawlab/rosdistro/hydro/rosdep.yaml" "/etc/ros/rosdep/sources.list.d/20-default.list"
chmod -R a+rX /etc/ros

rosdep update

rosdep install flymad --default-yes
rosmake flymad

chmod -R a+rX ${FLYMAD_TARGET}

