import roslib; roslib.load_manifest('flymad')
from flymad.msg import HeadDetect

PX = -0.6
PY = -0.6

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

