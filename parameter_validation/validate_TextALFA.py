from TextALFA_polygon.TextALFA_polygon import TextALFA
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from utils.polygon_operations import polygon_from_points
from eval_script_ic15.eval_script import polygon_evaluation_params


def validate_TextALFA(test_index, detector_keys, subm_dict, cur_params):
    text_alfa = TextALFA()
    evalParams = polygon_evaluation_params()
    joined_subm_dict = {}
    for img_key in test_index:
        joined_img_polygons = {}
        joined_img_confs = {}
        for j in range(len(detector_keys)):
            detector_key = detector_keys[j]
            detFile = subm_dict[detector_key][img_key]
            pointsList, _, confidencesList, _ = get_tl_line_values_from_file_contents(detFile, evalParams,
                                                                                      False, evalParams['CONFIDENCES'])

            detPols = []
            for i in range(len(pointsList)):
                points = pointsList[i]
                detPol = polygon_from_points(points)
                detPols.append(detPol)
            joined_img_polygons[detector_keys[j]] = detPols
            joined_img_confs[detector_keys[j]] = confidencesList
        new_img_polygon, new_img_confs = text_alfa.TextALFA_result(detector_keys, joined_img_polygons,
                                                                   joined_img_confs, tau=cur_params['tau'], gamma=0.5,
                                                                   bounding_box_fusion_method=cur_params['bounding_box_fusion_method'],
                                                                   scores_fusion_method=cur_params['scores_fusion_method'],
                                                                   add_empty_detections=True,
                                                                   empty_epsilon=cur_params['empty_epsilon'],
                                                                   max_1_box_per_detector=True,
                                                                   use_precision_instead_of_scores=cur_params['use_precision_instead_of_scores'])
        polygons, confs = [], []
        for i in range(len(new_img_polygon)):
            if new_img_confs[i] < cur_params['threshold']:
                continue
            polygons.append(new_img_polygon[i])
            confs.append(new_img_confs[i])
        new_img_polygon = polygons
        new_img_confs = confs

        joined_subm_dict[img_key] = new_img_polygon, new_img_confs
    return joined_subm_dict
