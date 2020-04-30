# (c) Evgeny Razinkov, Kazan Federal University, 2017
import numpy as np
from itertools import combinations
import cv2 as cv
import os
import json

from NMS.NMS_polygon import NMS_polygon
import TextALFA_polygon.clustering_polygon as polygon_clustering
from utils.polygon_operations import polygon_from_points, rectangle_to_polygon, order_points_new


class Object:
    def __init__(self, all_detectors_names, detectors_names, polygons, scores, bounding_box_fusion_method,
                 scores_fusion_method, add_empty_detections, empty_epsilon):
        self.bounding_box_fusion_method = bounding_box_fusion_method
        self.scores_fusion_method = scores_fusion_method
        self.add_empty_detections = add_empty_detections
        self.empty_epsilon = empty_epsilon
        self.detectors_names = all_detectors_names
        self.detected_by = detectors_names
        self.polygons = polygons
        self.scores = list(scores)
        self.epsilon = 0.0
        self.finalized = False

    def weighted_average(self):
        # for i in range(0, len(self.polygons)):
        #     self.polygons[i] = order_coordinates(self.polygons[i])
        points = np.asarray(self.polygons, dtype=np.int32)
        average_polygon = np.int0(np.average(points, axis=0, weights=self.np_scores[:len(self.polygons)])).flatten()
        return polygon_from_points(average_polygon)

    def intersection(self):
        # for i in range(0, len(self.polygons)):
        #     self.polygons[i] = order_coordinates(self.polygons[i])
        pInt = self.polygons[0]
        for i in range(1, len(self.polygons)):
            pInt = pInt & self.polygons[i]
        # if len(pInt) != 1:
        #     pInt = pInt | self.polygons[np.argmax(self.np_scores[:len(self.polygons)])]
        # pInt = order_coordinates(pInt)
        points = np.array(pInt, dtype=np.int32)[0]
        rect = cv.minAreaRect(points)
        rect = cv.boxPoints(rect)
        points = np.int0(rect).flatten()
        polygon = polygon_from_points(points)
        return polygon

    def most_confident_polygon(self):
        # for i in range(0, len(self.polygons)):
        #     self.polygons[i] = order_coordinates(self.polygons[i])
        most_confident = np.argmax(np.asarray(self.scores[:len(self.polygons)]))
        return self.polygons[most_confident]

    def average_polygon(self):
        # for i in range(0, len(self.polygons)):
        #     self.polygons[i] = order_coordinates(self.polygons[i])
        points = np.asarray(self.polygons, dtype=np.int32)
        average_polygon = np.int0(np.average(points, axis=0)).flatten()
        return polygon_from_points(average_polygon)

    def union(self):
        # for i in range(0, len(self.polygons)):
        #     self.polygons[i] = order_coordinates(self.polygons[i])
        pInt = self.polygons[0]
        for i in range(1, len(self.polygons)):
            pInt = pInt | self.polygons[i]
        points = np.array(pInt, dtype=np.int32)[0]
        rect = cv.minAreaRect(points)
        rect = cv.boxPoints(rect)
        points = np.int0(rect).flatten()
        polygon = polygon_from_points(points)
        return polygon

    def get_final_polygon(self):
        for i in range(0, len(self.polygons)):
            self.polygons[i] = order_points_new(self.polygons[i])
            # self.polygons[i] = order_coordinates(self.polygons[i])

        if self.bounding_box_fusion_method == 'INTERSECTION':
            return self.intersection()
        elif self.bounding_box_fusion_method == 'AVERAGE':
            return self.average_polygon()
        elif self.bounding_box_fusion_method == 'MOST_CONFIDENT':
            return self.most_confident_polygon()
        elif self.bounding_box_fusion_method == 'WEIGHTED_AVERAGE':
            return self.weighted_average()
        elif self.bounding_box_fusion_method == 'UNION':
            return self.union()
        else:
            print('Unknown value for bounding_box_fusion_method ' + self.bounding_box_fusion_method +
                  '. Using INTERSECTION')
            return self.intersection()

    def average_scores(self):
        return round(np.average(self.np_scores[:self.effective_scores], axis=0), 3)

    def multiply_scores(self):
        return round(np.prod(self.np_scores[:self.effective_scores]), 3)

    def get_final_score(self):
        if self.scores_fusion_method == 'AVERAGE':
            return self.average_scores()
        elif self.scores_fusion_method == 'MULTIPLY':
            return self.multiply_scores()
        elif self.scores_fusion_method == 'MOST_CONFIDENT':
            return max(self.scores[:self.effective_scores])
        else:
            print('Unknown value for class_scores_fusion_method ' + self.scores_fusion_method + '. Using AVERAGE')
            return self.average_scores()

    def finalize(self, detectors_names):
        self.detected_by_all = True
        for detector in detectors_names:
            if detector not in self.detected_by:
                self.detected_by_all = False
                self.scores.append(self.empty_epsilon)
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
            self.final_polygon = self.get_final_polygon()
            # self.final_polygon = get_convexhull(self.get_final_polygon())
            self.final_score = self.get_final_score()
            return self.final_polygon, self.final_score
        else:
            print('Zero objects, mate!')
        return None, None, None


class TextALFA:
    def __init__(self):
        self.bc = polygon_clustering.PolygonClustering()
        using_detections = [
            'psenet_015',
            'craft_015',
            'charnet_015',
            'psenet_93',
            'craft_85',
            'charnet_95',
        ]
        pr_curves_folder = './precision_recall/data/'
        self.curves = {}
        for key in using_detections:
            split = key.split('_')
            data_file = os.path.join(pr_curves_folder, split[0] + '_ic15_' + split[1] + '.json')
            with open(data_file, "r") as read_file:
                data = json.load(read_file)
            self.curves.update(data)


    def TextALFA_result(self, all_detectors_names, detectors_polygons, detectors_scores, tau, gamma,
                        bounding_box_fusion_method, scores_fusion_method, add_empty_detections, empty_epsilon,
                        max_1_box_per_detector, use_precision_instead_of_scores):
        """
        TextALFA algorithm

        ----------
        all_detectors_names : list
            Detectors names, that sholud have taken or have taken part in fusion. For e.g. ['ssd', 'denet', 'frcnn']
            even if 'ssd' didn't detect object.

        detectors_polygons : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's polygons.

            Example: {
            'craft': [[101, 254, 458, 255, 457, 658, 102, 658],
                    ...
                    [55, 21, 104, 25, 104, 396, 56, 397]],
            ...
            'charnet': [[201, 354, 468, 355, 467, 750, 202, 748],
                      ...
                      [553, 210, 678, 205, 670, 400, 550, 405]],
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

        use_precision_instead_of_scores: boolean
            True - replaces score with corresponding precision value of the detector
            False - leaves score unchanged

        Returns
        -------
        bounding_boxes : list
            Bounding boxes result of TextALFA

        scores : list
            Scores result of TextALFA
        """

        if use_precision_instead_of_scores:
            for key in detectors_scores:
                pr_data = self.curves[key]
                precision = pr_data['precision']
                thresholds = pr_data['thresholds']
                for i in range(len(detectors_scores[key])):
                    detectors_scores[key][i] = precision[thresholds.index(detectors_scores[key][i])]

        objects_detector_names, objects_polygons, objects_scores = \
            self.bc.get_raw_candidate_objects(detectors_polygons, detectors_scores, tau, gamma,
                                              max_1_box_per_detector)

        objects = []
        for i in range(0, len(objects_polygons)):
            objects.append(Object(all_detectors_names, objects_detector_names[i], objects_polygons[i],
                                  objects_scores[i], bounding_box_fusion_method, scores_fusion_method,
                                  add_empty_detections,
                                  empty_epsilon))

        polygons = []
        scores = []
        for detected_object in objects:
            object_polygon, object_score = \
                detected_object.get_object()
            if object_polygon is not None:
                polygons.append(object_polygon)
                scores.append(object_score)
        polygons = np.array(polygons)
        scores = np.array(scores)

        scores, polygons = NMS_polygon(scores, polygons)
        return polygons, scores
