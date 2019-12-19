# (c) Evgeny Razinkov, Kazan Federal University, 2017
import numpy as np

import bbox_clustering as bbox_clustering
from bbox_NMS import bbox_NMS


class Object:
    def __init__(self, all_detectors_names, detectors_names, bounding_boxes, angles, scores, bounding_box_fusion_method,
                 scores_fusion_method, add_empty_detections, empty_epsilon):
        self.bounding_box_fusion_method = bounding_box_fusion_method
        self.scores_fusion_method = scores_fusion_method
        self.add_empty_detections = add_empty_detections
        self.empty_epsilon = empty_epsilon
        self.detectors_names = all_detectors_names
        self.detected_by = detectors_names
        self.bounding_boxes = bounding_boxes
        self.angles = angles
        self.scores = list(scores)
        self.epsilon = 0.0
        self.finalized = False


    def max_bounding_box(self):
        lx = np.amin(self.np_bounding_boxes[:, 0])
        ly = np.amin(self.np_bounding_boxes[:, 1])
        rx = np.amax(self.np_bounding_boxes[:, 2])
        ry = np.amax(self.np_bounding_boxes[:, 3])
        return np.array([lx, ly, rx, ry])


    def min_bounding_box(self):
        lx = np.amax(self.np_bounding_boxes[:, 0])
        ly = np.amax(self.np_bounding_boxes[:, 1])
        rx = np.amin(self.np_bounding_boxes[:, 2])
        ry = np.amin(self.np_bounding_boxes[:, 3])
        return np.array([lx, ly, rx, ry])


    def average_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0)
        return bounding_box


    def weighted_average_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0, weights=self.np_scores[:len(self.np_bounding_boxes)])
        return bounding_box


    def get_final_bounding_box(self):
        if self.bounding_box_fusion_method == 'MAX':
            return self.max_bounding_box()
        elif self.bounding_box_fusion_method == 'MIN':
            return self.min_bounding_box()
        elif self.bounding_box_fusion_method == 'AVERAGE':
            return self.average_bounding_box()
        elif self.bounding_box_fusion_method == 'WEIGHTED AVERAGE':
            return self.weighted_average_bounding_box()
        elif self.bounding_box_fusion_method == 'MOST CONFIDENT':
            return self.np_bounding_boxes[np.argmax(self.np_scores)]
        else:
            print('Unknown value for bounding_box_fusion_method ' + self.bounding_box_fusion_method + '. Using AVERAGE')
            return self.average_bounding_box()


    def get_final_angle(self):
        return np.average(self.angles)


    def average_scores(self):
        return np.average(self.np_scores[:self.effective_scores], axis=0)


    def multiply_scores(self):
        temp = np.concatenate([self.np_scores[:self.effective_scores], 1 - self.np_scores[:self.effective_scores]], axis=1)
        temp = np.prod(np.clip(temp, a_min=self.epsilon, a_max=None), axis=0)
        return (temp / np.sum(temp))[:, 0]


    def get_final_score(self):
        if self.scores_fusion_method == 'AVERAGE':
            return self.average_scores()
        elif self.scores_fusion_method == 'MULTIPLY':
            return self.multiply_scores()
        elif self.scores_fusion_method == 'MOST CONFIDENT':
            return self.scores[np.argmax(self.scores)]
        else:
            print('Unknown value for class_scores_fusion_method ' + self.scores_fusion_method + '. Using AVERAGE')
            return self.average_scores()


    def finalize(self, detectors_names):
        self.detected_by_all = True
        for detector in detectors_names:
            if detector not in self.detected_by:
                self.detected_by_all = False
                self.scores.append(self.empty_epsilon)
        self.np_bounding_boxes = np.array(self.bounding_boxes)
        self.np_bounding_boxes = np.reshape(self.np_bounding_boxes,
                                            (len(self.bounding_boxes), len(self.bounding_boxes[0])))
        self.np_scores = np.array(self.scores)
        self.finalized = True


    def get_object(self):
        if not self.finalized:
            self.finalize(self.detectors_names)
        if len(self.scores) > 0:
            if self.add_empty_detections:
                self.effective_scores = len(self.np_scores)
            else:
                self.effective_scores = len(self.detected_by)
            self.final_bounding_box = self.get_final_bounding_box()
            self.final_angle = self.get_final_angle()
            self.final_score = self.get_final_score()
            return self.final_bounding_box, self.final_angle, self.final_score
        else:
            print('Zero objects, mate!')
        return None, None, None


class TextALFA:
    def __init__(self):
        self.bc = bbox_clustering.BoxClustering()


    def TextALFA_result(self, all_detectors_names, detectors_bounding_boxes, detectors_bounding_boxes_angles,
                    detectors_scores, tau, gamma, bounding_box_fusion_method, scores_fusion_method,
                    add_empty_detections, empty_epsilon, max_1_box_per_detector):
        """
        TextALFA algorithm

        ----------
        all_detectors_names : list
            Detectors names, that sholud have taken or have taken part in fusion. For e.g. ['ssd', 'denet', 'frcnn']
            even if 'ssd' didn't detect object.

        detectors_bounding_boxes : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's bounding boxes.

            Example: {
            'craft': [[10, 28, 128, 250],
                    ...
                    [55, 120, 506, 709]],
            ...
            'charnet': [[55, 169, 350, 790],
                      ...
                      [20, 19, 890, 620]],
            }


        detectors_bounding_boxes_angles : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's bounding boxes angles
            in radians.

            Example: {
            'craft': [3.14,
                    ...
                    1.57],
            ...
            'charnet': [1.04,
                      ...
                      0.52],
            }


        detectors_scores : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's scores vectors,
            corresponding to bounding boxes.

            Example: {
            'craft': [0.8,
                    ...
                    0.0001],
            ...
            'charnet': [0.4,
                      ...
                      0.0000005],
            }

        tau : float
            Parameter tau in the paper, between [0.0, 1.0]

        gamma : float
            Parameter gamma in the paper, between [0.0, 1.0]

        bounding_box_fusion_method : str
            Bounding box fusion method ["MIN", "MAX", "MOST CONFIDENT", "AVERAGE", "WEIGHTED AVERAGE",
            "WEIGHTED AVERAGE FINAL LABEL"]

        scores_fusion_method : str
            Scores fusion method ["MOST CONFIDENT", "AVERAGE", "MULTIPLY"]

        add_empty_detections : boolean
            True - low confidence class scores tuple will be added to cluster for each detector, that missed
            False - low confidence class scores tuple won't be added to cluster for each detector, that missed

        empty_epsilon : float
            Parameter epsilon in the paper, between [0.0, 1.0]

        max_1_box_per_detector : boolean
            True - only one detection form detector could be added to cluster
            False - multiple detections from same detector could be added to cluster

        Returns
        -------
        bounding_boxes : list
            Bounding boxes result of TextALFA

        scores : list
            Scores result of TextALFA
        """

        objects_detector_names, objects_boxes, objects_angles, objects_scores = \
            self.bc.get_raw_candidate_objects(detectors_bounding_boxes, detectors_bounding_boxes_angles,
                                              detectors_scores, tau, gamma, max_1_box_per_detector)

        objects = []
        for i in range(0, len(objects_boxes)):
            objects.append(Object(all_detectors_names, objects_detector_names[i], objects_boxes[i], objects_angles[i],
                       objects_scores[i], bounding_box_fusion_method, scores_fusion_method, add_empty_detections,
                                  empty_epsilon))

        bounding_boxes = []
        angles = []
        scores = []
        for detected_object in objects:
            object_bounding_box, object_angle, object_scores = \
                detected_object.get_object()
            if object_bounding_box is not None:
                bounding_boxes.append(object_bounding_box)
                angles.append(object_angle)
                scores.append(object_scores)
        bounding_boxes = np.array(bounding_boxes)
        scores = np.array(scores)

        bounding_boxes, angles, scores = bbox_NMS(scores, bounding_boxes, angles)
        return bounding_boxes, angles, scores


