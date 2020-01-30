from __future__ import division
from numpy import *

link = lambda a, b: concatenate((a, b[1:]))
edge = lambda a, b: concatenate(([a], [b]))


def qhull2D(sample):
    def dome(sample, base):
        h, t = base
        dists = dot(sample - h, dot(((0, -1), (1, 0)), (t - h)))
        outer = repeat(sample, dists > 0, 0)
        if len(outer):
            pivot = sample[argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:, 0]
        base = take(sample, [argmin(axis), argmax(axis)], 0)
        return link(dome(sample, base), dome(sample, base[::-1]))
    else:
        return sample
