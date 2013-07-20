from py2scad import *
import numpy as np

# this is based on py2scad examples/basic_enclosure.py

INCH2MM = 25.4

# Inside dimensions
x,y,z = 110.0, 70.0, 40
board_x = 88.5 # width of board
hole_list = []

# Create BNC holes in front
hole_left_x = x-63.15
dh = 33.26*0.5

offset_front_x0 = (x-board_x)*0.5

front_x0 = x*0.5 + offset_front_x0
front_y0 = z*0.5
for hole_x in [hole_left_x, hole_left_x+dh, hole_left_x+2*dh]:
    hole_y = 12.0
    hole = {
        'panel'     : 'front',
        'type'      : 'round',
        'location'  : (hole_x-front_x0,hole_y-front_y0),
        'size'      : 11.65,
        }
    hole_list.append(hole)

if 1:
    # mini-USB plug
    usb_hole_x = hole_left_x+2*dh + 17.8
    usb_hole_y = hole_y + 5.1
    hole = {
        'panel'     : 'front',
        'type'      : 'square',
        'location'  : (usb_hole_x-front_x0,usb_hole_y-front_y0),
        'size'      : (12.0, 10.0), # big enough for entire connector
        }
    hole_list.append(hole)

if 1:
    # power posts
    left_y0 = z*0.5

    dh = 14
    for hole_x in [-dh,0,dh]:
        hole_y = 24.0
        hole = {
            'panel'     : 'left',
            'type'      : 'round',
            'location'  : (hole_x,hole_y-left_y0),
            'size'      : 4.2,
            }
        hole_list.append(hole)

if 1:
    # load posts
    right_y0 = z*0.5

    dh = 14
    for hole_x in [-dh,0]:
        hole_y = 24.0
        hole = {
            'panel'     : 'right',
            'type'      : 'round',
            'location'  : (hole_x,hole_y-right_y0),
            'size'      : 4.2,
            }
        hole_list.append(hole)

if 1:
    # mounting holes
    dd = 25.0

    nx = 6
    ny = 2
    mx0 = -nx*dd/2
    my0 = -ny*dd/2

    if 0:
        print '*'*2000
        print 'MOUNTING HOLE DIAMETER IS FOR TESTING DO NOT MAKE THIS PART'
        print '*'*2000
        hole_diam = 11.0
    else:
        hole_diam = 7.0

    for hole_x in [mx0, mx0+nx*dd]:
        for hole_y in [my0,my0+ny*dd]:
            hole = {
                'panel'     : 'bottom',
                'type'      : 'round',
                'location'  : (hole_x,hole_y),
                'size'      : hole_diam,
                }
            hole_list.append(hole)


params = {
        'inner_dimensions'        : (x,y,z),
        'wall_thickness'          : 3.0,
        'lid_radius'              : 0.25*INCH2MM,
        'top_x_overhang'          : 0.2*INCH2MM,
        'top_y_overhang'          : 0.2*INCH2MM,
        'bottom_x_overhang'       : 0.2*INCH2MM + 20.0,
        'bottom_y_overhang'       : 0.2*INCH2MM,
        'lid2front_tabs'          : (0.2,0.5,0.8),
        'lid2side_tabs'           : (0.25, 0.75),
        'side2side_tabs'          : (0.5,),
        'lid2front_tab_width'     : 0.75*INCH2MM,
        'lid2side_tab_width'      : 0.75*INCH2MM,
        'side2side_tab_width'     : 0.5*INCH2MM,
        'standoff_diameter'       : 0.25*INCH2MM,
        'standoff_offset'         : 0.05*INCH2MM,
        'standoff_hole_diameter'  : 0.116*INCH2MM,
        'hole_list'               : hole_list,
        }

enclosure = Basic_Enclosure(params)
enclosure.make()

part_assembly = enclosure.get_assembly(explode=(5,5,5))
part_projection = enclosure.get_projection(show_ref_cube=False)

if 1:
    prog_assembly = SCAD_Prog()
    prog_assembly.fn = 50
    prog_assembly.add(part_assembly)
    prog_assembly.write('enclosure_assembly.scad')

prog_projection = SCAD_Prog()
prog_projection.fn = 50
prog_projection.add(part_projection)
prog_projection.write('enclosure_projection.scad')
