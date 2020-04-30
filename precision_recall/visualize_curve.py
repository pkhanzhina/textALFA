import matplotlib.pyplot as plt
import numpy as np
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from eval_script_ic15.eval_script import evaluate_method, polygon_evaluation_params, validate_data
from eval_script_ic15 import rrc_evaluation_funcs
from sklearn.metrics import precision_recall_curve
import json


# using_detections = [
#     'psenet_ic15_93',
#     'craft_ic15_85',
#     'charnet_ic15_95',
#     'nms_ic15_015',
#     'text_alfa_ic15_015'
# ]


using_detections = [
    'psenet_ic15_015',
    'craft_ic15_015',
    'charnet_ic15_015',
    'psenet_ic15_93',
    'craft_ic15_85',
    'charnet_ic15_95',
]

thresholds = {
    'psenet_ic15_015': None,
    'craft_ic15_015': None,
    'charnet_ic15_015': None,
    'psenet_ic15_93': 0.93,
    'craft_ic15_85': 0.85,
    'charnet_ic15_95': 0.95,
}


allDetFilePaths = {
    'craft_ic15_85': './res_craft_ic15/res_craft_ic15_85_2.zip',
    'craft_ic15_015': './res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_ic15_93': './res_psenet_ic15/res_psenet_ic15_93.zip',
    'psenet_ic15_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_ic15_95': './res_charnet_ic15/res_charnet_ic15_95.zip',
    'charnet_ic15_015': './res_charnet_ic15/res_charnet_ic15_015.zip',
    'nms_ic15_015': './res_nms_ic15/res_nms_ic15_psenet_015_craft_015_weighted_charnet_015.zip',
    'text_alfa_ic15_015': './res_text_alfa_polygon_ic15/res_text_alfa_polygon_ic15_psenet_015_craft_015_weighted_charnet_015.zip'
}

gtFilePath = './gt_ic15/gt_ic15.zip'


def precision_recall(gt, subm, threshold=None):
    evalParams = polygon_evaluation_params()
    p = {
        'g': gt,
        's': subm
    }
    resDict, precision, recall, confidences = rrc_evaluation_funcs.main_evaluation_notzip(p, evalParams, validate_data, evaluate_method,
                                                          show_result=False)
    operating_point = None
    if threshold is not None:
        op_p_idx = np.argmin(np.abs(confidences - threshold))
        operating_point = (float(precision[op_p_idx]), float(recall[op_p_idx]))
    return precision, recall, confidences, operating_point


def count_pr_for_detectors():
    evalParams = polygon_evaluation_params()
    validate_data(gtFilePath, evalParams, isGT=True)
    gt = load_zip_file(gtFilePath, evalParams['GT_SAMPLE_NAME_2_ID'])
    subm_dict = {}
    curves = {}
    for key in using_detections:
        validate_data(allDetFilePaths[key], evalParams, isGT=False)
        subm_dict[key] = load_zip_file(allDetFilePaths[key], evalParams['DET_SAMPLE_NAME_2_ID'], True)
        precision, recall, confidences, operating_point = precision_recall(gt, subm_dict[key], threshold=thresholds[key])
        curves[key] = {'precision': list(precision.astype(np.float)),
                           'recall': list(recall.astype(np.float)),
                           'thresholds': list(confidences),
                           'operating_point': operating_point}
        with open('./precision_recall/data/' + key + '.json', "w") as write_file:
            json.dump({key: curves[key]}, write_file)
    print()
    return curves


if __name__ == '__main__':
    curves = count_pr_for_detectors()
    # curves = {}
    # for name in using_detections:
    #     data_file = './precision_recall/data/' + name + '.json'
    #     with open(data_file, "r") as read_file:
    #         data = json.load(read_file)
    #     curves.update(data)
    plt.figure(figsize=(17, 10))
    for key in curves.keys():
        curves[key]['recall'].insert(0, 0.0)
        curves[key]['precision'].insert(0, max(curves[key]['precision']))
        plt.plot(curves[key]['recall'], curves[key]['precision'], label=key)
        if curves[key]['operating_point'] is None:
            continue
        plt.plot(curves[key]['operating_point'][1], curves[key]['operating_point'][0], 'xk', markersize=10)
        print(key, curves[key]['operating_point'])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.ylim(0.65, 1.05)
    plt.legend(loc=0)
    plt.show()
