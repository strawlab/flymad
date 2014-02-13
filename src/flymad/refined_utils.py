import numpy as np
import collections
import threading


import roslib; roslib.load_manifest('flymad')
from flymad.msg import HeadDetect

PX = -0.6
PY = -0.6

TTM_NAMES = {HeadDetect.TARGET_HEAD:"head",
             HeadDetect.TARGET_BODY:"body",
}

def target_dx_dy_from_message(msg):
    """returns None,None if the head/body was not detected"""

    dx = dy = None
    tx = ty = HeadDetect.NO_DETECT

    if msg.target_type == HeadDetect.TARGET_HEAD:
        tx = msg.head_x
        ty = msg.head_y
    elif msg.target_type == HeadDetect.TARGET_BODY:
        tx = msg.body_x
        ty = msg.body_y

    if tx != HeadDetect.NO_DETECT:
        dx = msg.target_x - tx
        dy = msg.target_y - ty

    return dx, dy

class StatsManager:
    def __init__(self, secs, fps=100):
        types = (HeadDetect.TARGET_HEAD, HeadDetect.TARGET_BODY)
        self._processingtime = {k:collections.deque(maxlen=secs*fps) for k in types}
        self._accuracy = {k:collections.deque(maxlen=secs*fps) for k in types}
        self._framenumber = 0
        self._lock = threading.Lock()

    def process(self, msg):
        if not (msg.framenumber > self._framenumber):
            return

        try:
            with self._lock:
                self._processingtime[msg.target_type].append(msg.processing_time)
                head_hit = msg.head_x != HeadDetect.NO_DETECT
                body_hit = msg.body_x != HeadDetect.NO_DETECT
                if msg.target_type == HeadDetect.TARGET_HEAD:
                    self._accuracy[HeadDetect.TARGET_HEAD].append(head_hit)
                self._accuracy[HeadDetect.TARGET_BODY].append(body_hit)
        except KeyError:
            pass

    def get_stats(self):
        with self._lock:
            pt = {k:np.mean(v) for k,v in self._processingtime.iteritems()}
            ac = {k:np.mean(v) for k,v in self._accuracy.iteritems()}
        return pt,ac

