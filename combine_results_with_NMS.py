import os
import numpy as np
from zipfile import ZipFile

from script_test.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, main_evaluation
from script_test.script import default_evaluation_params, validate_data, evaluate_method
from IoU_computation import polygon_from_points
from NMS import polygons_nms

allDetFilePaths = {
    'craft_015': './res_craft_ic15/res_craft_ic15_015.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip'
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
    detector_keys = list(subm_dict.keys())
    if len(detector_keys) == 0:
        exit()
    img_keys = list(subm_dict[detector_keys[0]].keys())
    joined_subm_dict = {}
    for img_key in img_keys:
        joined_img_polygons = []
        joined_img_confs = []
        for detector_key in detector_keys:
            detFile = subm_dict[detector_key][img_key]
            pointsList, confidencesList, _ = get_tl_line_values_from_file_contents(detFile,
                                            evalParams['CRLF'], evalParams['LTRB'], False, evalParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                detPol = polygon_from_points(points)
                joined_img_polygons.append(detPol)
                joined_img_confs.append(confidencesList[n])
        new_img_confs, new_img_polygons = polygons_nms(joined_img_confs, joined_img_polygons)
        joined_subm_dict[img_key] = (new_img_confs, new_img_polygons)
    full_output_dir = os.path.join('./res_nms_ic15', result_name)
    if not os.path.exists(full_output_dir):
        os.mkdir(full_output_dir)
    zip_filename = os.path.join(full_output_dir, result_name + '.zip')
    zipObj = ZipFile(zip_filename, 'w')
    for img_key in img_keys:
        new_img_confs, new_img_polygons = joined_subm_dict[img_key]
        output_filename = 'res_img_%s.txt' % img_key
        full_output_filename = os.path.join(full_output_dir, output_filename)
        with open(full_output_filename, 'w') as f:
            f.write('\n'.join([','.join(list(np.reshape(new_img_polygons[i][0], -1).astype('int32').astype('string')) \
                                        + [str(new_img_confs[i])])
             for i in range(len(new_img_confs))]))
        zipObj.write(output_filename)
    zipObj.close()
    p = {
        'g': gtFilePath,
        's': zip_filename
    }
    main_evaluation(p, evalParams, validate_data, evaluate_method, show_result=True, per_sample=True)