import os
import numpy as np
from zipfile import ZipFile

from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from eval_script_ic15.eval_script import default_evaluation_params, validate_data
from polygon_operations import polygon_from_points
from polygon_NMS import polygons_nms
from visualize_detections import visualize

allDetFilePaths = {
    'craft_015': './res_craft_ic15/res_craft_ic15_015_mean.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
}

gtFilePath = './gt_ic15/gt_ic15.zip'
detFilePaths = {
    'craft': allDetFilePaths['craft_015'],
    'psenet': allDetFilePaths['psenet_015']
}

result_name = 'res_nms_ic15_craft_015_mean_psenet_015'

if __name__ == '__main__':
    evalParams = default_evaluation_params()
    for key in detFilePaths:
        validate_data(detFilePaths[key], evalParams, isGT=False)
    subm_dict = {}
    for key in detFilePaths:
        subm_dict[key] = load_zip_file(detFilePaths[key], evalParams['DET_SAMPLE_NAME_2_ID'], True)
    validate_data(gtFilePath, evalParams, isGT=True)
    gt = load_zip_file(gtFilePath, evalParams['GT_SAMPLE_NAME_2_ID'])
    detector_keys = list(subm_dict.keys())
    if len(detector_keys) == 0:
        exit()
    img_keys = list(sorted(subm_dict[detector_keys[0]].keys(), key=lambda x: int(x)))
    joined_subm_dict = {}
    for img_key in img_keys:
        gtFile = decode_utf8(gt[img_key])
        gtPols = []
        gtConfs = []
        gtDontCare = []
        pointsList, _, transcriptionsList = get_tl_line_values_from_file_contents(gtFile, evalParams['CRLF'],
                                                                            evalParams['LTRB'], True, False)
        for i in range(len(pointsList)):
            points = pointsList[i]
            transcription = transcriptionsList[i]
            dontCare = transcription == "###"
            gtDontCare.append(dontCare)
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtConfs.append(1.0)
        img = visualize(img_key, gtConfs, gtPols, gtDontCare, 'gt')
        joined_img_polygons = []
        joined_img_confs = []
        for detector_key in detector_keys:
            detFile = subm_dict[detector_key][img_key]
            pointsList, confidencesList, _ = get_tl_line_values_from_file_contents(detFile,
                                            evalParams['CRLF'], evalParams['LTRB'], False, evalParams['CONFIDENCES'])
            detPols = []
            for i in range(len(pointsList)):
                points = pointsList[i]
                detPol = polygon_from_points(points)
                detPols.append(detPol)
                joined_img_polygons.append(detPol)
                joined_img_confs.append(confidencesList[i])
            visualize(img_key, confidencesList, detPols, [False] * len(detPols), detector_key, img)
        new_img_confs, new_img_polygons = polygons_nms(joined_img_confs, joined_img_polygons)
        visualize(img_key, new_img_confs, new_img_polygons, [False] * len(new_img_confs), 'nms', img)
        joined_subm_dict[img_key] = (new_img_confs, new_img_polygons)
    zip_filename = os.path.join('./res_nms_ic15', result_name + '.zip')
    with ZipFile(zip_filename, 'w') as zipped_f:
        for img_key in img_keys:
            new_img_confs, new_img_polygons = joined_subm_dict[img_key]
            output_filename = 'res_img_%s.txt' % img_key
            zipped_f.writestr(output_filename,
                '\n'.join([','.join(list(np.reshape(new_img_polygons[i][0], -1).astype('int32').astype('string'))) \
                                        for i in range(len(new_img_confs))])) #+ [str(new_img_confs[i])])
    # p = {
    #     'g': gtFilePath,
    #     's': zip_filename
    # }
    # main_evaluation(p, evalParams, validate_data, evaluate_method, show_result=True, per_sample=True)