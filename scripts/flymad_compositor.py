#!/usr/bin/env python
import flydra.a2.benu as benu
import motmot.FlyMovieFormat.FlyMovieFormat as fmf
import argparse
import collections
import numpy as np
import motmot.imops.imops as imops
import warnings
import pytz
import datetime

import roslib; roslib.load_manifest('rosbag')
import rospy
import rosbag

def scale(w, h, x, y, maximum=True):
    # see http://code.activestate.com/recipes/577575-scale-rectangle-while-keeping-aspect-ratio/
    nw = y * w / h
    nh = x * h / w
    if maximum ^ (nw >= x):
        return nw or 1, y
    return x, nh or 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wide', type=str)
    parser.add_argument('zoom', type=str)
    parser.add_argument('rosbag', type=str)

    parser.add_argument('--wide_transform', type=str, default=None)
    parser.add_argument('--zoom_transform', type=str, default=None)
    args = parser.parse_args()

    doit(widef=args.wide,zoomf=args.zoom,rosbagf=args.rosbag,
         widet=args.wide_transform,
         zoomt=args.zoom_transform,
         )

class DateFormatter:
    def __init__(self,tz):
        self.tz = tz

    def format_date(self, x, pos=None):
        return str(datetime.datetime.fromtimestamp(x,self.tz))

def doit(widef=None,zoomf=None,rosbagf=None,
         widet=None, zoomt=None):
    wide = fmf.FlyMovie(widef)
    zoom = fmf.FlyMovie(zoomf)
    bag = rosbag.Bag(rosbagf)

    wide_ts = wide.get_all_timestamps()
    zoom_ts = zoom.get_all_timestamps()

    objs = collections.defaultdict(list)
    for topic, msg, t in bag.read_messages(topics='/flymad/tracked'):
        stamp = msg.header.stamp
        state = msg.state_vec
        objs[ msg.obj_id ] .append( (stamp.secs + stamp.nsecs*1e-9, state[0], state[1]) )

    obj_times = []
    for obj_id in objs:
        objs[ obj_id ] = np.array( objs[ obj_id ], dtype=[('stamp',np.float64),
                                                          ('x',np.float32),
                                                          ('y',np.float32)] )

        stamps = objs[obj_id]['stamp']
        obj_times.append( (np.min(stamps),
                           np.max(stamps),
                           obj_id ) )
    obj_times = np.array(obj_times)

    raw2d = []
    for topic, msg, t in bag.read_messages(topics='/flymad/raw_2d_positions'):
        stamp = msg.header.stamp
        stamp = stamp.secs + stamp.nsecs*1e-9
        for pt in msg.points:
            raw2d.append(( stamp, pt.x, pt.y ))

    raw2d = np.array( raw2d, dtype=[('stamp',np.float64),
                                     ('x',np.float32),
                                     ('y',np.float32)] )


    tzname = None
    for topic, msg, t in bag.read_messages(topics='/timezone'):
        tzname = msg.value
    if tzname is None:
        # default timezone
        tzname = 'CET'
        warnings.warn('No data in /timezone topic - setting default timezone '
                      'to %s'%tzname)

    start_time = np.max( [np.min(wide_ts),
                          np.min(zoom_ts),
                          np.min(raw2d['stamp'])] )
    stop_time = np.min( [np.max(wide_ts),
                         np.max(zoom_ts),
                         np.max(raw2d['stamp'])] )

    dur = stop_time-start_time
    FPS = 50.0
    rate = 1.0/FPS

    times = np.arange(start_time, stop_time+1e-10, rate)
    print 'at %f fps, this is %d frames'%(FPS,len(times))

    tz = pytz.timezone( tzname )
    pretty_time = DateFormatter(tz)

    for out_fno, cur_time in enumerate(times):
        print 'frame %d of %d'%(out_fno+1, len(times))
        valid_obj_cond = (obj_times[:,0] <= cur_time) & (cur_time <= obj_times[:,1])
        valid_obj_ids = map(int,obj_times[valid_obj_cond,2])

        wide_frame, this_wide_ts = wide.get_frame_at_or_before_timestamp(cur_time)
        zoom_frame, this_zoom_ts = zoom.get_frame_at_or_before_timestamp(cur_time)

        wide_frame = imops.auto_convert(wide.format, wide_frame)
        zoom_frame = imops.auto_convert(zoom.format, zoom_frame)

        #cond = (cur_time - rate < raw2d['stamp']) & (raw2d['stamp'] <= cur_time)
        cond = (cur_time - 50*rate < raw2d['stamp']) & (raw2d['stamp'] <= cur_time)
        this_raw2d = raw2d[cond]

        save_fname_path = 'out%06d.png'%out_fno
        final_w = 1024
        final_h = 768

        margin = 10
        max_panel_w = final_w // 2 - 3*margin//2
        max_panel_h = final_h - 2*margin

        canv=benu.Canvas(save_fname_path, final_w, final_h)

        # wide-angle view --------------------------
        x0 = 0; w = wide.get_width()
        y0 = 0; h = wide.get_height()
        if 1:
            warnings.warn('WARNING: zooming on center region as a hack!!!!')
            x0+=100; w-=240
        user_rect = (x0,y0,w,h)
        if widet is not None and '90' in widet: # rotate aspect ratio
            w,h = h,w
        dev_w, dev_h = scale( w, h, max_panel_w, max_panel_h )
        device_rect = (margin, margin, dev_w, dev_h)


        r = 25
        wz_height = max_panel_h//2
        dev_w, dev_h = scale( r*2, r*2, max_panel_w, wz_height )
        device_rect_crop = (margin, final_h - margin - wz_height, dev_w, dev_h)

        if 1:
            warnings.warn( 'magnified wide-field view should use objs[ obj_id ] to zoom on tracked object')

        xc = this_raw2d['x'][-1]
        yc = this_raw2d['y'][-1]

        x0 = xc-r
        x1 = xc+r
        if x0 < 0:
            x0 = 0
            x1 = 2*r
        elif x1 >= wide.get_width():
            x1 = wide.get_width-1
            x0 = x1-2*r

        y0 = yc-r
        y1 = yc+r
        if y0 < 0:
            y0 = 0
            y1 = 2*r
        elif y1 >= wide.get_height():
            y1 = wide.get_height()-1
            y0 = y1-2*r

        user_rect_crop = (x0,y0,2*r,2*r)
        for d,u in [ (device_rect, user_rect),
                     (device_rect_crop, user_rect_crop) ]:
            with canv.set_user_coords(d, u, transform=widet):
                canv.imshow(wide_frame,0,0,filter='nearest')
                canv.scatter( this_raw2d['x'],
                              this_raw2d['y'],
                              color_rgba=(0,1,0,0.3), radius=2.0 )
                for obj_id in valid_obj_ids:
                    stamps = objs[ obj_id ]['stamp']
                    cond = stamps <= cur_time
                    last_idx = np.nonzero(cond)[0][-1]
                    r = objs[obj_id][last_idx]
                    canv.scatter( [r['x']], [r['y']],
                                  color_rgba=(1,0,1,0.4), radius=1.5 )
                    canv.text( '%d' % obj_id,
                               r['x'], r['y'],
                               color_rgba=(1,0,1,0.4))


        # zoomed view --------------------------
        w,h = zoom.get_width(), zoom.get_height()
        if zoomt is not None and '90' in zoomt: # rotate aspect ratio
            w,h = h,w
        dev_w, dev_h = scale( w, h, max_panel_w, max_panel_h )
        device_rect = (final_w // 2 + margin//2, margin, dev_w, dev_h)
        user_rect = (0,0, zoom.get_width(), zoom.get_height())
        with canv.set_user_coords(device_rect, user_rect, transform=zoomt):
            canv.imshow(zoom_frame,0,0,filter='best')

        canv.text( '%s' % pretty_time.format_date(cur_time),
                   5,15,
                   color_rgba=(1,1,1,1))

        canv.save()

if __name__=='__main__':
    main()
