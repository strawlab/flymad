import json
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches

import shapely.geometry as sg

import madplot

dat = json.load(open('conf.json'))
arena = madplot.Arena(dat)

control = os.path.join(dat['_base'],dat['control'])

ax = plt.figure().add_subplot(111)

ldf,tdf,geom = madplot.load_bagfile(control)

madplot.plot_tracked_trajectory(ax, tdf, intersect_patch=arena.get_intersect_patch(geom, fill=True, color='r', closed=True))
madplot.plot_geom(ax, geom)


#circle = sg.Point(360,260).buffer(200)
#poly = sg.Polygon(list(zip(*geom)))
#inter = circle.intersection(poly)

#pat = matplotlib.patches.Polygon(list(inter.exterior.coords), fill=True, color='r', closed=True)
#print pat
#print arena.get_intersect_patch(geom, fill=True, color='r', closed=True)
##ax.add_patch(pat)

plt.show()

