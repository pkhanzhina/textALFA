from eval_script_ic15.rrc_evaluation_funcs import load_zip_file, get_tl_line_values_from_file_contents, decode_utf8
from eval_script_ic15.eval_script import default_evaluation_params, validate_data

do_visualization = False
using_detections = [
    'psenet_015',
    'craft_015_weighted',
    'charnet_015'
]

allDetFilePaths = {
    'craft_015': './res_craft_ic15/res_craft_ic15_015.zip',
    'craft_015_mean': './res_craft_ic15/res_craft_ic15_015_mean.zip',
    'craft_015_weighted': './res_craft_ic15/res_craft_ic15_015_weighted.zip',
    'psenet_015': './res_psenet_ic15/res_psenet_ic15_015.zip',
    'charnet_015': './res_charnet_ic15/res_charnet_ic15_015.zip'
}
gtFilePath = './gt_ic15/gt_ic15.zip'
result_name = 'res_nms_ic15_' + '_'.join(using_detections)


if __name__ == '__main__':
    evalParams = default_evaluation_params()
    evalParams['LTRB'] = True
    evalParams['ANGLE'] = True