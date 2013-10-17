import threading

import roslib
roslib.load_manifest('flymad')

import rospy
import geometry_msgs.msg
import std_msgs.msg

import cv2
import numpy as np

class _HitManager(object):
    def __init__(self, target_when_inside):
        if target_when_inside:
            self.should_target = self.is_inside_area
        else:
            self.should_target = lambda x,y: not self.is_inside_area(x,y)

class HitBox(_HitManager):
    def __init__(self, target_when_inside, borders_x, borders_y):
        _HitManager.__init__(self, target_when_inside)
        self._borders_x = map(int,borders_x)
        self._borders_y = map(int,borders_y)

    def is_inside_area(self, x, y):
        return  ((self._borders_x[0] <= x <= self._borders_x[1]) and
                 (self._borders_y[0] <= y <= self._borders_y[1]))

class HitImage(_HitManager):
    def __init__(self, target_when_inside, change_cb=None):
        _HitManager.__init__(self, target_when_inside)
        self._change_cb = change_cb
        self._img_lock = threading.Lock()
        self._img = None

        self._origin = 'top left'
        rospy.Subscriber('/draw_geom/image_origin',
                        std_msgs.msg.String,
                        self._on_image_origin)
        #width,height
        self._size = tuple()
        rospy.Subscriber('/draw_geom/image_size',
                        geometry_msgs.msg.Point32,
                        self._on_image_size)
        self._poly = []
        rospy.Subscriber('/draw_geom/poly',
                        geometry_msgs.msg.Polygon,
                        self._on_image_poly)

    def _on_image_origin(self, msg):
        self._origin = msg.data
        self._maybe_update()

    def _on_image_size(self, msg):
        self._size = (msg.x, msg.y)
        self._maybe_update()

    def _on_image_poly(self, msg):
        self._poly = msg.points
        self._maybe_update()

    def _maybe_update(self):
        if not self._size:
            return
        if not self._origin:
            return
        if not self._poly:
            return

        with self._img_lock:
            img = np.zeros((self._size[1], self._size[0]), dtype=np.uint8)
            poly = []
            for pt in self._poly:
                poly.append( (pt.x,pt.y) )
            cv2.fillPoly(img,[np.array(poly, np.int32)],255)
            self._img = img

        if self._change_cb:
            self._change_cb(self)

    @property
    def image(self):
        if self._img is None:
            return None

        with self._img_lock:
            return self._img.copy()

    def is_inside_area(self, x, y):
        if self._img is None:
            return False
        try:
            with self._img_lock:
                return self._img[y,x]
        
        except IndexError:
            return False

