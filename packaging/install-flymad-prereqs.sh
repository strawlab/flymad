#!/bin/bash -x
set -e

# ----- Check that we have the supported version of Ubuntu ----------
RELEASE=`lsb_release --release | cut -f2`
if [ ${RELEASE} == "12.04" ]; then
    echo "OK: using Ubuntu 12.04"
else
    echo "ERROR: release ${RELEASE} not supported"
    exit 1
fi

ARCH=`uname -i`
if [ ${ARCH} == "x86_64" ]; then
    echo "OK: using adm64 architecture"
else
    echo "ERROR: architecture ${ARCH} not supported"
    exit 1
fi

# ----- Check that we have root permissions ----

if [ "$UID" -ne "0" ]; then
    echo "ERROR: you need to have superuser permissions. Re-run with 'sudo'."
    exit 1
fi

# ----- Update our packages ------------

apt-get update --yes
apt-get install --yes python-software-properties

# -------------- add our repository -----------
#     add key id=F8DB323D
wget -qO - http://debs.strawlab.org/astraw-archive-keyring.gpg | sudo apt-key add -
add-apt-repository 'deb http://debs.strawlab.org/ precise/'

# -------------- add ROS repository -----------
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
add-apt-repository 'deb http://packages.ros.org/ros/ubuntu precise main'

# -------------- update local package index ----
apt-get update --yes

# ---- install ROS

DEBIAN_FRONTEND=noninteractive apt-get install --yes python-rosinstall ros-hydro-rosmake ros-hydro-ros-base ros-hydro-geometry-msgs ros-hydro-sensor-msgs make
#DEBIAN_FRONTEND=noninteractive apt-get install --yes python-rosinstall ros-hydro-rosmake ros-hydro-desktop-full
