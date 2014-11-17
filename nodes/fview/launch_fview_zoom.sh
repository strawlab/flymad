#!/bin/sh

if ifconfig eth2 | grep -q MTU:9000 ; then
  echo "FVIEW RUNNING WITH JUMBO FRAMES - GOOD"
  LIBCAMIFACE_ARAVIS_PACKET_SIZE=8000 python /usr/bin/fview __name:=fview_ttm --plugins=fview_head_track
else
  echo "FVIEW RUNNING WITHOUT JUMBO FRAMES - BAD"
  echo "you should type"
  echo "sudo ifconfig eth2 mtu 9000 up"
  python /usr/bin/fview __name:=fview_ttm --plugins=fview_head_track
fi

