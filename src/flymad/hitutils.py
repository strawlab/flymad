import threading

import roslib
roslib.load_manifest('flymad')

import rospy
import geometry_msgs.msg
import std_msgs.msg

import cv2
import numpy as np

class HitBox:
    def __init__(self, borders_x, borders_y):

        self.in_area = (lambda x,y:
                ((int(borders_x[0]) <= x <= int(borders_x[1])) and
                 (int(borders_y[0]) <= y <= int(borders_y[1]))) )

class HitImage(object):
    def __init__(self, change_cb=None):
        self._change_cb = change_cb
        self._img_lock = threading.Lock()
        self._img = None

        self._origin = 'top left'
        rospy.Subscriber('/flyman/geom_image_origin',
                        std_msgs.msg.String,
                        self._on_image_origin)
        #width,height
        self._size = tuple()
        rospy.Subscriber('/flyman/geom_image_size',
                        geometry_msgs.msg.Point32,
                        self._on_image_size)
        self._poly = []
        rospy.Subscriber('/flymad/geom_poly',
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

    def in_area(self, x, y):
        if self._img is None:
            return False
        try:
            with self._img_lock:
                return self._img[y,x]
        
        except IndexError:
            return False

