import os
from zipfile import ZipFile
import numpy as np

from TextALFA_polygon import TextALFA
from eval_script_ic15.eval_script import polygon_evaluation_params, validate_data
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from utils.visualize_detections import visualize_polygons
from utils.polygon_operations import polygon_from_points

do_visualization = True
using_detections = [
    # 'psenet_015',
    # 'craft_015_weighted',
    'charnet_015'
]

allDetFilePaths = {
    'craft_015_weighted': './res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_015': './res_charnet_ic15/res_charnet_ic15_015.zip'
}
gtFilePath = './gt_ic15/gt_ic15.zip'
result_name = 'res_text_alfa_polygon_ic15_' + '_'.join(using_detections)


if __name__ == '__main__':
    text_alfa = TextALFA()
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
    joined_subm_dict = {}
    img_keys = list(sorted(subm_dict[detector_keys[0]].keys(), key=lambda x: int(x)))
    joined_subm_dict = {}
    for img_key in img_keys:
        gtFile = decode_utf8(gt[img_key])
        gtPols = []
        gtConfs = []
        gtDontCare = []
        pointsList, _, _, transcriptionsList = get_tl_line_values_from_file_contents(gtFile, evalParams, True, False)
        for i in range(len(pointsList)):
            transcription = transcriptionsList[i]
            dontCare = transcription == "###"
            gtDontCare.append(dontCare)
            points = pointsList[i]
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtConfs.append(1.0)
        if do_visualization:
            img = visualize_polygons(img_key, gtConfs, gtPols, gtDontCare, 'gt')
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
            if do_visualization:
                visualize_polygons(img_key, confidencesList, detPols, [False] * len(pointsList), detector_key)
        new_img_polygon, new_img_confs = text_alfa.TextALFA_result(detector_keys, joined_img_polygons,
                                                                joined_img_confs, tau=0.1, gamma=0.5,
                                                                bounding_box_fusion_method='INTERSECTION',
                                                                scores_fusion_method='AVERAGE',
                                                                add_empty_detections=True,
                                                                empty_epsilon=0.1,
                                                                max_1_box_per_detector=True)
        if do_visualization:
            visualize_polygons(img_key, new_img_confs, new_img_polygon,
                                        [False] * len(new_img_confs), 'text_alfa', img, show=True)
        joined_subm_dict[img_key] = new_img_polygon, new_img_confs
    zip_filename = os.path.join('./res_text_alfa_polygon_ic15', result_name + '.zip')
    with ZipFile(zip_filename, 'w') as zipped_f:
        for img_key in img_keys:
            new_img_polygons, new_img_confs = joined_subm_dict[img_key]
            output_filename = 'res_img_%s.txt' % img_key
            zipped_f.writestr(output_filename,
                '\n'.join([','.join(list(np.reshape(new_img_polygons[i][0], -1).astype('int32').astype('string')) +
                                    [str(new_img_confs[i])]) for i in range(len(new_img_confs))]))