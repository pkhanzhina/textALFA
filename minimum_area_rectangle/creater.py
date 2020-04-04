import os
import numpy as np
from zipfile import ZipFile
from minimum_area_rectangle.min_bounding_rect import *
from minimum_area_rectangle.qhull_2d import *


using_detections = [
    'psenet_015',
    'craft_015_weighted',
    'charnet_015',
    'gt_ic15'
]

allDetFilePaths = {
    'craft_015': '../res_craft_ic15/res_craft_ic15_015.zip',
    'craft_015_mean': '../res_craft_ic15/res_craft_ic15_015_mean.zip',
    'craft_015_weighted': '../res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_015': '../res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_015': '../res_charnet_ic15/res_charnet_ic15_015.zip',
    'gt_ic15': '../gt_ic15/gt_ic15.zip'
}
detector = using_detections[3]

z_name = allDetFilePaths[detector]
new_z_name = '' + detector + '_rect.zip'
z = ZipFile(z_name, 'r')
new_z = ZipFile(new_z_name, 'w')

for filename in z.namelist():
    file = z.open(filename, "r")
    s = ""
    for line in file:
        n, line = len(line), str(line)
        old = line[2:(n)].split(',')
        score = old[8]
        print(filename)
        c = np.asarray(old[:8], dtype=np.int).reshape((4, 2))
        c = np.concatenate([c, [c[0]]])
        hull_points = qhull2D(c)
        hull_points = hull_points[::-1]
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
        xtl, ytl = int(corner_points[2, 0]), int(corner_points[2, 1])
        xbr, ybr = int(corner_points[0, 0]), int(corner_points[0, 1])
        temp = str(xtl) + "," + str(ytl) + "," + str(xbr) + "," + str(ybr) + "," + str(rot_angle) + "," + score + '\n'
        s = s + temp
    new_z.writestr(filename, s)
    file.close()
z.close()
new_z.close()
