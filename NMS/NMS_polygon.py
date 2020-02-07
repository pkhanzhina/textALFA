# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Additional Numpy methods. Big mess of many things!
"""

import numpy as np

from utils.polygon_operations import get_intersection_over_union


def NMS_polygon(scores, polygons, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    sorted_ind = np.argsort(scores)[::-1]
    scores = [scores[idx] for idx in sorted_ind]
    polygons = [polygons[idx] for idx in sorted_ind]
    keep_bboxes = np.ones(len(scores), dtype=np.bool)
    for i in range(len(scores)-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            keep_overlap = []
            for idx in range(i + 1, len(polygons)):
                keep_overlap.append(get_intersection_over_union(polygons[i], polygons[idx]) < nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)[0]
    return [scores[idx] for idx in idxes], [polygons[idx] for idx in idxes]