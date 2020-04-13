import Polygon as plg
# from shapely.geometry import Polygon
import numpy as np
from imutils import perspective
from scipy.spatial import ConvexHull


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)

    pointMat = resBoxes[0].reshape([2, 4]).T

    return plg.Polygon(pointMat)


def rectangle_to_points(rect):
    points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin),
              int(rect.xmin), int(rect.ymin)]
    return points


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0.0


def order_coordinates(polygon):
    points = np.array(polygon, dtype=np.int32)[0]
    ordered = perspective.order_points(points)
    return polygon_from_points((np.int0(ordered)).flatten())


def get_convexhull(polygon):
    points = np.array(polygon)[0]
    hull = (ConvexHull(points=points))
    points = points[hull.vertices]
    return polygon_from_points((np.int0(points)).flatten())


def order_points_new(polygon):
    pts = np.array(polygon, dtype=np.int32)[0]
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    ordered = np.array([tl, tr, br, bl], dtype="float32")
    return polygon_from_points((np.int0(ordered)).flatten())