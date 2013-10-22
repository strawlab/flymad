#!/bin/sh

if ifconfig eth2 | grep -q MTU:9000 ; then
  echo "FVIEW RUNNING WITH JUMBO FRAMES - GOOD"
  LIBCAMIFACE_ARAVIS_PACKET_SIZE=8000 fview --plugins=5
else
  echo "FVIEW RUNNING WITHOUT JUMBO FRAMES - BAD"
  echo "you should type"
  echo "sudo ifconfig eth2 mtu 9000 up"
  fview
fi

