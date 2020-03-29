import matplotlib.pyplot as plt
import numpy as np
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from eval_script_ic15.eval_script import evaluate_method, polygon_evaluation_params, validate_data
from eval_script_ic15 import rrc_evaluation_funcs
from sklearn.metrics import precision_recall_curve

using_detections = [
    'psenet_015',
    'craft_015_weighted',
    'charnet_015',
    'nms_ic15'
]

allDetFilePaths = {
    'craft_015_weighted': './res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_015': './res_charnet_ic15/res_charnet_ic15_015.zip',
    'nms_ic15': './res_nms_ic15/res_nms_ic15_psenet_015_craft_015_weighted_charnet_015.zip'
}
gtFilePath = './gt_ic15/gt_ic15.zip'


def precision_recall(img_keys, gt, subm):
    precision = []
    recall = []
    for th in np.arange(0.01, 1.001, 0.001):
        new_subm = {}
        for key in img_keys:
            new_subm[key] = []
            detFile = decode_utf8(subm[key])
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
    # indices = sorted(range(len(recall)), key=lambda i: recall[i])
    # recall = np.array(recall)
    # precision = np.array(precision)
    # recall = recall[indices]
    # precision = precision[indices]
    return precision, recall


if __name__ == '__main__':
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
        curves[new_key] = precision_recall(img_keys, gt, subm_dict[new_key])
    for key in curves.keys():
        plt.plot(curves[key][1], curves[key][0], label=key)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc=3)
    plt.show()

    # precision, recall, thresholds = precision_recall_curve(gt, subm_dict[new_key])
