import os
import numpy as np

from script_test.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from script_test.script import default_evaluation_params, validate_data
from IoU_computation import polygon_from_points
from NMS import polygons_nms
from visualize_detections import visualize

allDetFilePaths = {
    'craft_015': './res_craft_ic15/res_craft_ic15_015.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
}

gtFilePath = './gt_ic15/gt_ic15.zip'
detFilePaths = {
    'craft': allDetFilePaths['craft_015'],
    'psenet': allDetFilePaths['psenet_015']
}

result_name = 'res_nms_ic15_craft_015_psenet_015'

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
    img_keys = list(subm_dict[detector_keys[0]].keys())
    joined_subm_dict = {}
    for img_key in img_keys:
        gtFile = decode_utf8(gt[img_key])
        gtPols = []
        gtConfs = []
        pointsList, _, _ = get_tl_line_values_from_file_contents(gtFile, evalParams['CRLF'],
                                                                            evalParams['LTRB'], True, False)
        for i in range(len(pointsList)):
            points = pointsList[i]
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtConfs.append(1.0)
        # img = visualize(img_key, gtConfs, gtPols, 'gt')
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
            # visualize(img_key, confidencesList, detPols, detector_key, img)
        new_img_confs, new_img_polygons = polygons_nms(joined_img_confs, joined_img_polygons)
        # visualize(img_key, new_img_confs, new_img_polygons, 'nms', img)
        joined_subm_dict[img_key] = (new_img_confs, new_img_polygons)
    full_output_dir = os.path.join('./res_nms_ic15', result_name)
    if not os.path.exists(full_output_dir):
        os.mkdir(full_output_dir)
    # zip_filename = os.path.join(full_output_dir, result_name + '.zip')
    # zipObj = ZipFile(zip_filename, 'w')
    for img_key in img_keys:
        new_img_confs, new_img_polygons = joined_subm_dict[img_key]
        output_filename = 'res_img_%s.txt' % img_key
        full_output_filename = os.path.join(full_output_dir, output_filename)
        with open(full_output_filename, 'w') as f:
            f.write('\n'.join([','.join(list(np.reshape(new_img_polygons[i][0], -1).astype('int32').astype('string')) \
                                        + [str(new_img_confs[i])])
             for i in range(len(new_img_confs))]))
        # zipObj.write(output_filename)
    # zipObj.close()
    # p = {
    #     'g': gtFilePath,
    #     's': zip_filename
    # }
    # main_evaluation(p, evalParams, validate_data, evaluate_method, show_result=True, per_sample=True)