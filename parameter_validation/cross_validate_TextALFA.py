from sklearn.model_selection import KFold
import os
from zipfile import ZipFile
import numpy as np
import random

from eval_script_ic15.eval_script import evaluate_method
from eval_script_ic15 import rrc_evaluation_funcs
from eval_script_ic15.eval_script import polygon_evaluation_params, validate_data
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file
from parameter_validation.validate_TextALFA import validate_TextALFA


using_detections = [
    'psenet_015',
    'craft_015_weighted',
    'charnet_015'
]

allDetFilePaths = {
    'craft_015_weighted': './res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_015': './res_charnet_ic15/res_charnet_ic15_015.zip'
}
gtFilePath = './gt_ic15/gt_ic15.zip'
bounding_box_fusion_method = ["AVERAGE", "INTERSECTION", "MOST_CONFIDENT", "WEIGHTED_AVERAGE", "UNION"]
scores_fusion_method = ["AVERAGE", "MULTIPLY", "MOST_CONFIDENT"]

folds_count = 10

all_results = [{} for i in range(10)]


def generate_params():
    # params = {'tau': [], 'bounding_box_fusion_method': [], 'scores_fusion_method': [], 'empty_epsilon': []}
    params = {}
    # random.seed(1e6-1)
    params['tau'] = round(random.uniform(0.01, 1), 2)
    params['empty_epsilon'] = round(random.uniform(0.01, 1), 2)
    params['bounding_box_fusion_method'] = bounding_box_fusion_method[random.randint(0, 4)]
    params['scores_fusion_method'] = scores_fusion_method[random.randint(0, 2)]
    params['threshold'] = random.uniform(0.015**3, 0.5)
    # params['threshold'] = random.uniform(0.015**3, 0.015)
    # params['threshold'] = random.uniform(0.015, 0.5)
    return params


def cross_validate_TextALFA(img_keys, gt, detector_keys, subm_dict, folds_count=folds_count):
    img_keys = np.array(img_keys)

    # random.seed(1e6-1)
    random_indices = random.sample(range(len(img_keys)), len(img_keys))
    img_keys = img_keys[random_indices]
    kf = KFold(n_splits=folds_count)

    fold_index = 0
    best_params_per_fold = {i: {'tau': 0.00,
                                'bounding_box_fusion_method': "",
                                'scores_fusion_method': "",
                                'empty_epsilon': 0.00} for i in range(folds_count)}
    best_result_per_fold = {i: {'precision': 0.00,
                                'recall': 0.00,
                                'hmean': 0.00,
                                'AP': 0.00} for i in range(folds_count)}
    fold_index = 0
    for _, test_index in kf.split(img_keys):
        img_keys_fold = img_keys[test_index]

        gt_name = 'gt_ic15_' + str(fold_index)
        gt_zip_filename = os.path.join('./parameter_validation/data', gt_name + '.zip')
        with ZipFile(gt_zip_filename, 'w') as zipped_f:
            for key in img_keys_fold:
                output_filename = 'gt_img_%s.txt' % key
                zipped_f.writestr(output_filename, gt[key])
        fold_index += 1

    for k in range(0, 10000):
        cur_params = generate_params()
        print()
        print(k, cur_params)
        fold_index = 0
        for _, test_index in kf.split(img_keys):
            img_keys_fold = img_keys[test_index]
            result_name = 'res_cross_validation_' + str(fold_index)
            joined_subm_dict = validate_TextALFA(img_keys_fold, gt, detector_keys, subm_dict, cur_params)
            gt_name = 'gt_ic15_' + str(fold_index)
            gt_zip_filename = os.path.join('./parameter_validation/data', gt_name + '.zip')
            zip_filename = os.path.join('./parameter_validation/data', result_name + '.zip')

            with ZipFile(zip_filename, 'w') as zipped_f:
                for key in img_keys_fold:
                    new_img_polygons, new_img_confs = joined_subm_dict[key]
                    output_filename = 'res_img_%s.txt' % key
                    zipped_f.writestr(output_filename,
                                      '\n'.join(
                                          [','.join(list(
                                              np.reshape(new_img_polygons[i][0], -1).astype('int32').astype(
                                                  'str')) +
                                                    [str(new_img_confs[i])]) for i in
                                           range(len(new_img_confs))]))

            p = {
                'g': gt_zip_filename,
                's': zip_filename
            }
            resDict = rrc_evaluation_funcs.main_evaluation(p, evalParams, validate_data, evaluate_method, show_result=False)
            all_results[fold_index][str(cur_params)] = resDict
            if resDict['AP'] > best_result_per_fold[fold_index]['AP'] and \
                    resDict['precision'] > best_result_per_fold[fold_index]['precision']:  # and \
                    # resDict['recall'] > best_result_per_fold[fold_index]['recall'] and \
                    # resDict['hmean'] > best_result_per_fold[fold_index]['hmean']:
                best_result_per_fold[fold_index] = resDict
                best_params_per_fold[fold_index] = cur_params
            fold_index += 1
        av_ap = 0
        av_recall = 0
        av_precision = 0
        av_hmean = 0
        for i in range(0, folds_count):
            print("Fold number: ", i, " AP: ", round(best_result_per_fold[i]['AP'] * 100, 2))
            av_ap += best_result_per_fold[i]['AP']
            av_recall += best_result_per_fold[i]['recall']
            av_precision += best_result_per_fold[i]['precision']
            av_hmean += best_result_per_fold[i]['hmean']
        av_ap /= folds_count
        av_recall /= folds_count
        av_precision /= folds_count
        av_hmean /= folds_count
        print("average precision: ", av_precision, " average recall: ", av_recall, " average hmean: ", av_hmean, " average ap: ", av_ap)
    # ap_per_fold = []
    # for i in range(0, folds_count):
    #     params = best_params_per_fold[i]
    #     sum = 0
    #     for j in range(0, folds_count):
    #         sum += all_results[j][str(params)]['AP']
    #     ap_per_fold.append(sum / folds_count)
    print()
    print('Cross parameter validation results:')
    av_ap = 0
    av_recall = 0
    av_precision = 0
    av_hmean = 0
    for i in range(0, folds_count):
        print('Fold number: ', i, '   Average precision: ', round(best_result_per_fold[i]['AP'] * 100, 2))
        print("Params: ", best_params_per_fold[i])
        av_ap += best_result_per_fold[i]['AP']
        av_recall += best_result_per_fold[i]['recall']
        av_precision += best_result_per_fold[i]['precision']
        av_hmean += best_result_per_fold[i]['hmean']
        # print("AP/folds_count: ", ap_per_fold[i])
        print()
    av_ap /= folds_count
    av_recall /= folds_count
    av_precision /= folds_count
    av_hmean /= folds_count
    print("average precision: ", av_precision)
    print("average recall: ", av_recall)
    print("average hmean: ", av_hmean)
    print("average ap: ", av_ap)
    return best_result_per_fold, best_params_per_fold


if __name__ == '__main__':
    evalParams = polygon_evaluation_params()

    for key in using_detections:
        validate_data(allDetFilePaths[key], evalParams, isGT=False)
    subm_dict = {}
    for key in using_detections:
        new_key = key.split('_')[0]
        subm_dict[new_key] = load_zip_file(allDetFilePaths[key], evalParams['DET_SAMPLE_NAME_2_ID'], True)
    validate_data(gtFilePath, evalParams, isGT=True)
    gt = load_zip_file(gtFilePath, evalParams['GT_SAMPLE_NAME_2_ID'])
    detector_keys = list(subm_dict.keys())
    if len(detector_keys) == 0:
        exit()
    img_keys = list(sorted(subm_dict[detector_keys[0]].keys(), key=lambda x: int(x)))
    best_results, best_params = cross_validate_TextALFA(img_keys, gt, detector_keys, subm_dict)
    print()
    print()
    for i in range(0, folds_count):
        print(i, best_results[i])
        print(best_params[i])

