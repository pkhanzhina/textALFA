# (c) Evgeny Razinkov, Kazan Federal University, 2017

import numpy as np


def fastIoU(boxes):
    number_of_boxes = len(boxes)
    m0 = np.zeros((number_of_boxes, number_of_boxes, 4))
    m0[:] = boxes
    m0T = np.transpose(m0, axes = (1, 0, 2))
    m_up_left = np.maximum(m0[:, :, :2], m0T[:, :, :2])
    m_down_right = np.minimum(m0[:, :, 2:], m0T[:, :, 2:])
    m_diff = np.clip(m_down_right - m_up_left, a_min = 0.0, a_max=None)
    m_intersection = m_diff[:, :, 0] * m_diff[:, :, 1]
    m_area = (m0[:, :, 2] - m0[:, :, 0]) * (m0[:, :, 3] - m0[:, :, 1])
    m_T_area = (m0T[:, :, 2] - m0T[:, :, 0]) * (m0T[:, :, 3] - m0T[:, :, 1])
    iou = m_intersection / (m_area + m_T_area - m_intersection)
    return iou


def fastDiff(detector_indices):
    number_of_boxes = len(detector_indices)
    s0 = np.zeros((number_of_boxes, number_of_boxes))
    s0[:] = detector_indices
    s_diff = np.not_equal(s0 - s0.T, 0.0).astype(float)
    return s_diff


def fastAngleSame(angles):
    number_of_boxes = len(angles)
    a0 = np.zeros((number_of_boxes, number_of_boxes))
    a0[:] = angles
    a_diff_1 = np.abs(a0 - a0.T)
    a_diff_2 = 2 * np.pi - a_diff_1
    a_diff = np.where(a_diff_1 < a_diff_2, a_diff_1, a_diff_2)
    a_sim = np.cos(a_diff)
    return a_sim


class BoxClustering:
    def __init__(self):
        pass

    def prepare_matrix(self):
        self.sim_matrix = fastIoU(self.boxes)
        if self.max_1_box_per_detector:
            s_diff = fastDiff(self.names)
            self.sim_matrix = self.sim_matrix * (s_diff +  + np.eye(self.n_boxes))
        a_sim = fastAngleSame(self.angles)
        self.sim_matrix = np.power(self.sim_matrix, self.power_iou) * np.power(a_sim, 1.0 - self.power_iou)
        self.path_matrix = np.greater_equal(self.sim_matrix, self.hard_threshold).astype(int)


    def get_paths(self):
        self.whole_path_matrix = self.path_matrix
        new_paths_might_exist = True
        while new_paths_might_exist:
            self.new_whole_path_matrix = np.matmul(self.whole_path_matrix, np.transpose(self.whole_path_matrix))
            self.new_whole_path_matrix = np.greater_equal(self.new_whole_path_matrix, 0.5).astype(int)
            if np.array_equal(self.whole_path_matrix, self.new_whole_path_matrix):
                new_paths_might_exist = False
            self.whole_path_matrix = self.new_whole_path_matrix


    def cluster_indices(self, indices):
        clusters = [[a] for a in indices]
        lc = len(clusters)
        if lc == 1:
            return clusters
        else:
            ci1, ci2, sim = self.find_clusters_to_merge(clusters)
            while sim > self.hard_threshold:
                clusters = self.merge_clusters(ci1, ci2, clusters)
                if len(clusters) == 1:
                    return clusters
                else:
                    ci1, ci2, sim = self.find_clusters_to_merge(clusters)
            return clusters

        
    def find_clusters_to_merge(self, clusters):
        max_cluster_sim = 0.0
        ci1 = None
        ci2 = None
        if len(clusters) < 2:
            return 0, 0, 1.0
        for i in range(1, len(clusters)):
            for j in range(0, i):
                cl_sim = self.cluster_distance(clusters[i], clusters[j])
                if max_cluster_sim < cl_sim:
                    ci1 = i
                    ci2 = j
                    max_cluster_sim = cl_sim
        return ci1, ci2, max_cluster_sim


    def merge_clusters(self, ci1, ci2, clusters):
        if ci1 == ci2:
            return clusters
        clusters[ci1] = clusters[ci1] + clusters[ci2]
        del clusters[ci2]
        return clusters


    def cluster_distance(self, c1, c2):
        min_sim_init = False
        min_sim = 0.0
        for x1 in c1:
            for x2 in c2:
                if not min_sim_init:
                    min_sim = self.sim_matrix[x1, x2]
                    min_sim_init = True
                else:
                    if min_sim > self.sim_matrix[x1, x2]:
                        min_sim = self.sim_matrix[x1, x2]
        return min_sim


    def get_raw_candidate_objects(self, bounding_boxes, angles, scores, tau, gamma, max_1_box_per_detector):
        """
        Clusters detections from different detectors.

        ----------
        bounding_boxes : dict
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

        angles : dict
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

        scores : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's class scores vectors,
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

        same_labels_only : boolean
            True - only detections with same class label will be added into same cluster
            False - detections labels won't be taken into account while clustering

        use_BC : boolean
            True - Bhattacharyya and Jaccard coefficient will be used to compute detections similarity score
            False - only Jaccard coefficient will be used to compute detections similarity score

        max_1_box_per_detector : boolean
            True - only one detection form detector could be added to cluster
            False - multiple detections from same detector could be added to cluster


        Returns
        -------
        objects_boxes : list
            List containing [clusters, cluster bounding boxes, box coordinates]
            cluster bounding boxes - bounding boxes added to cluster

        objects_detector_names : list
            List containing [clusters, cluster detectors names]
            cluster detectors names - names of detectors, corresponding to bounding boxes added to cluster

        objects_class_scores : list
            List containing [clusters, cluster class scores]
            cluster class scores - class scores, corresponding to bounding boxes added to cluster
        """

        self.hard_threshold = tau
        self.power_iou = gamma
        self.max_1_box_per_detector = max_1_box_per_detector

        self.detector_names = list(bounding_boxes.keys())

        self.n_boxes = 0
        self.names = np.zeros(0)
        self.boxes = np.zeros((0, 4))
        self.angles = np.zeros(0)
        self.scores = np.zeros(0)

        detector_index = 0
        self.actual_names = []
        for detector_name in self.detector_names:
            detector_boxes = len(bounding_boxes[detector_name])
            if detector_boxes > 0:
                self.n_boxes += detector_boxes
                self.boxes = np.vstack((self.boxes, bounding_boxes[detector_name]))
                self.actual_names += [detector_name] * detector_boxes
                self.names = np.hstack((self.names, np.ones(detector_boxes) * detector_index))
                self.angles = np.vstack((self.angles, angles[detector_name]))
                self.scores = np.vstack((self.scores, scores[detector_name]))
            detector_index += 1

        self.prepare_matrix()

        self.get_paths()

        objects_detector_names = []
        objects_boxes = []
        objects_scores = []
        objects_angles = []
        np_detectors = np.array(self.actual_names)
        np_boxes = np.array(self.boxes)
        np_angles = np.array(self.angles)
        if len(self.whole_path_matrix):
            unique_whole_path_matrix = np.vstack({tuple(row) for row in self.whole_path_matrix})
            for i in range(0, len(unique_whole_path_matrix)):
                indices, = np.where(unique_whole_path_matrix[i] > 0)
                if len(indices) > 0:
                    clusters = self.cluster_indices(list(indices))
                    for c in clusters:
                        objects_detector_names.append(list(np_detectors[c]))
                        objects_boxes.append(np_boxes[c])
                        objects_angles.append(np_angles[c])
                        objects_scores.append(scores[c])

        return objects_detector_names, objects_boxes, objects_angles, objects_scores

