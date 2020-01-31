from TextALFA_bbox import TextALFA
from eval_script_ic15.eval_script import box_with_angle_evaluation_params, validate_data
from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from utils.visualize_detections import visualize_boxes_with_angles

do_visualization = True
using_detections = [
    'psenet_015',
    'craft_015_weighted',
    'charnet_015'
]

allDetFilePaths = {
    'craft_015_weighted': './res_craft_ic15/res_craft_ic15_015_weighted_rect.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015_rect.zip',
    'charnet_015': './res_charnet_ic15/res_charnet_ic15_015_rect.zip'
}
gtFilePath = './gt_ic15/gt_ic15_rect.zip'
result_name = 'res_text_alfa_ic15_' + '_'.join(using_detections) + '_rect'


if __name__ == '__main__':
    text_alfa = TextALFA()
    evalParams = box_with_angle_evaluation_params()
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
        gtBoxes = []
        gtAngles = []
        gtConfs = []
        gtDontCare = []
        pointsList, anglesList, _, transcriptionsList = get_tl_line_values_from_file_contents(gtFile, evalParams, True,
                                                                                              False)
        for i in range(len(pointsList)):
            transcription = transcriptionsList[i]
            dontCare = transcription == "###"
            gtDontCare.append(dontCare)
            gtBoxes.append(pointsList[i])
            gtAngles.append(anglesList[i])
            gtConfs.append(1.0)
        if do_visualization:
            img = visualize_boxes_with_angles(img_key, gtConfs, gtBoxes, gtAngles, gtDontCare, 'gt')
        joined_img_boxes = {}
        joined_img_angles = {}
        joined_img_confs = {}
        for j in range(len(detector_keys)):
            detector_key = detector_keys[j]
            detFile = subm_dict[detector_key][img_key]
            pointsList, anglesList, confidencesList, _ = get_tl_line_values_from_file_contents(detFile, evalParams,
                                                                                     False, evalParams['CONFIDENCES'])
            joined_img_boxes[detector_key] = pointsList
            joined_img_angles[detector_key] = anglesList
            joined_img_confs[detector_key] = confidencesList
            if do_visualization:
                show = False
                if (j + 1) == len(detector_keys):
                    show = True
                visualize_boxes_with_angles(img_key, confidencesList, pointsList, anglesList, [False] * len(pointsList),
                                            detector_key, show=show)
        # new_img_boxes, new_img_angles, new_img_confs = text_alfa.TextALFA_result(detector_keys, joined_img_boxes,
        #                                                         joined_img_angles, joined_img_confs, tau=0.1, gamma=0.5,
        #                                                         bounding_box_fusion_method='WEIGHTED AVERAGE',
        #                                                         scores_fusion_method='AVERAGE',
        #                                                         add_empty_detections=True,
        #                                                         empty_epsilon=0.1, use_angle=False,
        #                                                         max_1_box_per_detector=True)
        # if do_visualization:
        #     visualize_boxes_with_angles(img_key, new_img_confs, new_img_boxes, new_img_angles,
        #                                 [False] * len(new_img_confs), 'text_alfa', img, show=True)
        # joined_subm_dict[img_key] = new_img_boxes, new_img_angles, new_img_confs
    # zip_filename = os.path.join('./res_text_alfa_ic15', result_name + '.zip')
    # with ZipFile(zip_filename, 'w') as zipped_f:
    #     for img_key in img_keys:
    #         new_img_boxes, new_img_angles, new_img_confs = joined_subm_dict[img_key]
    #         output_filename = 'res_img_%s.txt' % img_key
    #         zipped_f.writestr(output_filename,
    #                           '\n'.join([','.join(
    #                               list(np.reshape(new_img_boxes[i], -1).astype('int32').astype('string')) +
    #                               [str(new_img_angles[i])] +
    #                               [str(new_img_confs[i])]) for i in range(len(new_img_confs))]))