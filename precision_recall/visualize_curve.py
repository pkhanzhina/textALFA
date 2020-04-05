import matplotlib.pyplot as plt
import numpy as np
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from eval_script_ic15.eval_script import evaluate_method, polygon_evaluation_params, validate_data
from eval_script_ic15 import rrc_evaluation_funcs
from sklearn.metrics import precision_recall_curve
import json


using_detections = [
    'psenet_ic15_93',
    'craft_ic15_85',
    'charnet_ic15_95',
    'nms_ic15',
    'textalfa_ic15'
]

allDetFilePaths = {
    'craft_ic15_85': './res_craft_ic15/res_craft_ic15_85.zip',
    'psenet_ic15_93': './res_psenet_ic15/res_psenet_ic15_93.zip',
    'charnet_ic15_95': './res_charnet_ic15/res_charnet_ic15_95.zip',
    'nms_ic15': './res_nms_ic15/res_nms_ic15_psenet_015_craft_015_charnet_015.zip',
    'textalfa_ic15': './res_text_alfa_polygon_ic15/res_text_alfa_polygon_ic15_psenet_015_craft_015_charnet_015.zip'
}

thresholds = {
    'psenet': 0.93,
    'craft': 0.85,
    'charnet': 0.95,
    'nms': None,
    'textalfa': None
}

gtFilePath = './gt_ic15/gt_ic15.zip'


def precision_recall(img_keys, gt, subm, flag=False, threshold=None):
    operating_point = None
    evalParams = polygon_evaluation_params()
    precision = []
    recall = []
    for th in np.arange(0.01, 1.001, 0.001):
        th = round(th, 3)
        new_subm = {}
        for key in img_keys:
            new_subm[key] = []
            if not flag:
                detFile = decode_utf8(subm[key])
            else:
                detFile = subm[key]
            # print(subm[key])
            # print(detFile)
            pointsList, _, confidencesList, _ = get_tl_line_values_from_file_contents(detFile, evalParams,
                                                                                      False, evalParams['CONFIDENCES'])

            new_detFile = '\n'.join([', '.join(
                (list(np.array(pointsList[i]).astype('int32').astype('str')) + [str(confidencesList[i])]) if
                confidencesList[i] > th else "") for i in range(len(confidencesList))])
            new_subm[key] = new_detFile.encode()

        p = {
            'g': gt,
            's': new_subm
        }
        resDict = rrc_evaluation_funcs.main_evaluation_notzip(p, evalParams, validate_data, evaluate_method,
                                                              show_result=False)
        subm = new_subm
        precision.append(resDict['precision'])
        recall.append(resDict['recall'])
        if th == threshold:
            operating_point = (precision[-1], recall[-1])
    precision.reverse()
    recall.reverse()
    return precision, recall, operating_point


def count_pr_for_detectors():
    evalParams = polygon_evaluation_params()
    validate_data(gtFilePath, evalParams, isGT=True)
    gt = load_zip_file(gtFilePath, evalParams['GT_SAMPLE_NAME_2_ID'])
    subm_dict = {}
    curves = {}
    for key in using_detections:
        validate_data(allDetFilePaths[key], evalParams, isGT=False)
        new_key = key.split('_')[0]
        subm_dict[new_key] = load_zip_file(allDetFilePaths[key], evalParams['DET_SAMPLE_NAME_2_ID'], True)
        img_keys = list(sorted(subm_dict[new_key].keys(), key=lambda x: int(x)))
        print(new_key)
        precision, recall, operating_point = precision_recall(img_keys, gt, subm_dict[new_key], threshold=thresholds[new_key])
        print(recall)
        print(operating_point)
        curves[new_key] = {'precision': precision,
                           'recall': recall,
                           'operating_point': operating_point}
    #     with open('./precision_recall/data/' + key + '.json', "w") as write_file:
    #         json.dump({new_key: curves[new_key]}, write_file)
    # print()
    return curves


# data_file = './precision_recall/data/pr_curve_015.json'


if __name__ == '__main__':
    # curves = count_pr_for_detectors()
    curves = {}
    for name in using_detections:
        data_file = './precision_recall/data/' + name + '.json'
        with open(data_file, "r") as read_file:
            data = json.load(read_file)
        curves.update(data)
    plt.figure(figsize=(17, 10))
    for key in curves.keys():
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
