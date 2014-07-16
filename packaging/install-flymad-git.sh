#!/bin/bash -x
set -e

# -----------------------------------------------------------------
#
# This program is called by the FlyMAD installer to install FlyMAD.
#
# It does the following:
#
#   1) calls the script ./install-flymad-prereqs.sh to ensure that
#      the machine is setup properly
#   2) it generates a .rosinstall spec file specifying the ROS
#      stacks necessary
#   3) it calls rosinstall to install these ROS stacks.
#
# -----------------------------------------------------------------

# Install as root user. (Do not let ROS install things into ~/.ros .)
export HOME=/root
export USER=root

./install-flymad-prereqs.sh

ROSINSTALL_SPEC_PATH="/tmp/flymad.rosinstall"
THISDIR=`pwd`

# ---- create a .rosinstall spec file -------------
cat > ${ROSINSTALL_SPEC_PATH} <<EOF
- git: {local-name: flymad, uri: 'https://github.com/strawlab/flymad', version: flymad-hydro}
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
  sudo rosdep init
fi

./ensure_line.py "yaml https://raw.github.com/strawlab/rosdistro/hydro/rosdep.yaml" "/etc/ros/rosdep/sources.list.d/20-default.list"
chmod -R a+rX /etc/ros

rosdep update

rosdep install flymad --default-yes
rosmake flymad

chmod -R a+rX ${FLYMAD_TARGET}

sudo desktop-file-install ${FLYMAD_TARGET}/flymad/packaging/gflymad.desktop
