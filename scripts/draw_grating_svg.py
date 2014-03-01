#!/usr/bin/env python
from __future__ import division
import argparse

import numpy as np

PAGE_WIDTH = 210
PAGE_HEIGHT = 297

# per row/column
DEFAULT_WIDTH = PAGE_WIDTH

class Element(object):
    def __init__(self, x0, y0, w=10.0, h=10.0):
        self.x0=x0
        self.y0=y0
        self.w = w
        self.h = h
    def get_rects(self):
        result = []
        opts = dict(fill = "black",
                    )

        rd = dict(x=self.x0,
                  y=self.y0,
                  width=self.w,
                  height=self.h,
                  )
        rd.update(opts)
        return [rd]


def format_tag(tag,opts):
    attrs = []
    for k in opts:
        v = opts[k]
        attrs.append( k+'="'+str(v)+'"')
    attrs = ' '.join(attrs)
    elem = '<'+tag+' '+attrs+'/>\n'
    return elem

def make_svg_elements( rects=None ):
    if rects is None:
        rects = []

    out = ""

    for rect in rects:
        out += format_tag('rect',rect)
    return out

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mm', help='size of single check',
                        default=None, type=float)
    parser.add_argument('--pdf', help='use inkscape to convert to PDF',
                        default=False, action='store_true')
    args = parser.parse_args()

    width = DEFAULT_WIDTH

    if args.mm is None:
        wavelength= 30.0
        print "Wavelength not specified, setting default value."
    else:
        wavelength = args.mm

    print "Wavelength is %s"%(wavelength,)

    n_rows = int(np.ceil(PAGE_HEIGHT/wavelength))

    total_width = width
    total_height = n_rows*wavelength
    print 'total_width %g mm'%total_width
    print 'total_height %g mm'%total_height

    rect0 = dict(x=0,y=0, width=total_width, height=total_height, fill="white")
    rects = [rect0]

    # elements
    for i in range(n_rows):
        if i%2==0:
            rects.extend( Element(0,i*wavelength,w=width,h=wavelength).get_rects() )

    file_contents = """<?xml version="1.0"?>
<svg width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}"
     xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny">
{elements}
</svg>
"""

    elements = make_svg_elements( rects=rects )

    s=file_contents.format( w=total_width,
                            h=total_height,
                            elements=elements,
                          )
    fname_svg = 'square_grating_%s.svg'%(wavelength,)
    with open(fname_svg,mode='w') as fd:
        fd.write(s)
    print 'saved',fname_svg

    if args.pdf:
        import subprocess

        fname_pdf = fname_svg + '.pdf'

        cmd = 'inkscape -f %s --export-pdf=%s'%(fname_svg, fname_pdf)
        subprocess.check_call(cmd,shell=True)
        print 'saved',fname_pdf
