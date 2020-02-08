from zipfile import ZipFile
import numpy as np
import cv2 as cv

using_detections = [
    'psenet_ic15_015',
    'craft_ic15_015_weighted',
    'charnet_ic15_015',
    'gt_ic15'
]

allDetFilePaths = {
    'craft_ic15_015_weighted': '../res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_ic15_015': '../res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_ic15_015': '../res_charnet_ic15/res_charnet_ic15_015.zip',
    'gt_ic15': '../gt_ic15/gt_ic15.zip'
}
detector = using_detections[0]

z_name = allDetFilePaths[detector]
if detector != 'gt_ic15':
    new_z_name = 'res_' + detector + '_rect2.zip'
else:
    new_z_name = detector + '_rect2.zip'
z = ZipFile(z_name, 'r')
new_z = ZipFile(new_z_name, 'w')

for filename in z.namelist():
    file = z.open(filename, "r")
    s = ""
    for line in file:
        line = (line.decode()).replace(' ', '')
        old = line.split(',')
        score = old[8]
        print(filename)
        c = np.asarray(old[:8], dtype=np.int).reshape((4, 2))
        rect = cv.minAreaRect(c)
        center_point, size, rot_angle = rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        xtl, ytl = int(box[0, 0]), int(box[0, 1])
        xbr, ybr = int(box[2, 0]), int(box[2, 1])
        # print(c)
        # print(box, rot_angle+90)
        # print(size)
        rot_angle = np.deg2rad(rot_angle+90)
        temp = str(xtl) + "," + str(ytl) + "," + str(xbr) + "," + str(ybr) + "," + str(rot_angle) + "," + str(score)
        s = s + temp
    new_z.writestr(filename, s.encode())
    file.close()
z.close()
new_z.close()
