#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple

import eval_script_ic15.rrc_evaluation_funcs as rrc_evaluation_funcs
from utils.polygon_operations import *


def polygon_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
                'IOU_CONSTRAINT' :0.5,
                'AREA_PRECISION_CONSTRAINT' :0.5,
                'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
                'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
                'LTRB':False, #LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
                'CRLF':False, # Lines are delimited by Windows CRLF format
                'ANGLE':False, # Detections must include angle for bounding box
                'CONFIDENCES':True, # Detections must include confidence value. AP will be calculated
                'PER_SAMPLE_RESULTS':True #Generate per sample results and produce data for visualization
            }


def box_with_angle_evaluation_params():
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': True,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'ANGLE': True,  # Detections must include angle for bounding box
        'CONFIDENCES': True,  # Detections must include confidence value. AP will be calculated
        'PER_SAMPLE_RESULTS': True,  # Generate per sample results and produce data for visualization
        'ANGLE_CONSTRAINT': 2 * np.pi
    }



def validate_data(filePath, evaluationParams, isGT):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    if isGT:
        gt = rrc_evaluation_funcs.load_zip_file(filePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
        # Validate format of GroundTruth
        for k in gt:
            rrc_evaluation_funcs.validate_lines_in_file(k, gt[k], evaluationParams, withTranscription=True)
    else:
        subm = rrc_evaluation_funcs.load_zip_file(filePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
        #Validate format of results
        for k in subm:
            # if (k in gt) == False :
            #     raise Exception("The sample %s not present in GT" %k)
            rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams,
                                                        withConfidence=evaluationParams['CONFIDENCES'])


def compute_ap(confList, matchList, numGtCare, careMask=None):
    correct = 0
    AP = 0
    prec = []
    rec = []
    confs = []
    if len(confList) > 0:
        confList = np.array(confList)
        matchList = np.array(matchList)
        sorted_ind = np.argsort(-confList)
        confList = confList[sorted_ind]
        matchList = matchList[sorted_ind]
        tps = np.cumsum(np.array(matchList).astype(np.float32))
        fps = np.cumsum(np.array(np.logical_not(matchList)).astype(np.float32))
        prec = tps / np.maximum((tps + fps), np.finfo(np.float64).eps)
        rec = tps / np.maximum(float(numGtCare), np.finfo(np.float64).eps)
        confs = confList
        if careMask is not None:
            careMask = np.array(careMask)
            careMask = careMask[sorted_ind]
            confList = confList[careMask]
            matchList = matchList[careMask]
        for n in range(len(confList)):
            match = matchList[n]
            if match:
                correct += 1
                AP += float(correct) / (n + 1)

        if numGtCare > 0:
            AP /= numGtCare

    return AP, prec, rec, confs

    
def evaluate_method(gt, subm, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
    
    perSampleMetrics = {}
    
    matchedSum = 0
    
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
   
    numGlobalCareGt = 0;
    numGlobalCareDet = 0;
    
    arrGlobalConfidences = [];
    arrGlobalMatches = [];
    arrGlobalCareMask = [];

    total_count = 0

    for resFile in gt:
        
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0    
        
        detMatched = 0
        
        iouMat = np.empty([1,1])
        
        gtPols = []
        detPols = []

        gtAngles = []
        detAngles = []
        
        gtPolPoints = []
        detPolPoints = []  
        
        #Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        #Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []   
        
        pairs = [] 
        detMatchedNums = []
        
        arrSampleConfidences = [];
        arrSampleMatch = [];
        sampleAP = 0;

        evaluationLog = ""
        
        pointsList,anglesList,_,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams,True,False)
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            if evaluationParams['ANGLE']:
                gtAngles.append(anglesList[n])
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append( len(gtPols)-1 )
                
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")
        
        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
            
            pointsList,anglesList,confidencesList,_ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams,False,evaluationParams['CONFIDENCES'])
            total_count += len(pointsList)
            for n in range(len(pointsList)):
                points = pointsList[n]
                
                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)                    
                detPols.append(detPol)
                if evaluationParams['ANGLE']:
                    detAngles.append(anglesList[n])
                detPolPoints.append(points)
                if len(gtDontCarePolsNum)>0 :
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break
                                
            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")
            
            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols),len(detPols)]
                iouMat = np.empty(outputShape)
                angleMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols),np.int8)
                detRectMat = np.zeros(len(detPols),np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)
                        if evaluationParams['ANGLE']:
                            angle_diff_1 = np.abs(gtAngles[gtNum] - detAngles[detNum])
                            angle_diff_2 = 2 * np.pi - angle_diff_1
                            angleMat[gtNum, detNum] = min(angle_diff_1, angle_diff_2)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            iou_pass = iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']
                            angle_pass = True
                            if evaluationParams['ANGLE']:
                                angle_pass = angleMat[gtNum, detNum] < evaluationParams['ANGLE_CONSTRAINT']
                            if iou_pass and angle_pass:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                pairs.append({'gt':gtNum,'det':detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    #we exclude the don't care detections
                    arrGlobalCareMask.append(detNum not in detDontCarePolsNum)

                    match = detNum in detMatchedNums

                    arrSampleConfidences.append(confidencesList[detNum])
                    arrSampleMatch.append(match)

                    arrGlobalConfidences.append(confidencesList[detNum])
                    arrGlobalMatches.append(match)
                            
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
            sampleAP = precision
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare==0 else float(detMatched) / numDetCare
            if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
                sampleAP, _, _, _ = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)

        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)                

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        
        if evaluationParams['PER_SAMPLE_RESULTS']:
            perSampleMetrics[resFile] = {
                                            'precision':precision,
                                            'recall':recall,
                                            'hmean':hmean,
                                            'pairs':pairs,
                                            'AP':sampleAP,
                                            'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
                                            'gtPolPoints':gtPolPoints,
                                            'detPolPoints':detPolPoints,
                                            'gtDontCare':gtDontCarePolsNum,
                                            'detDontCare':detDontCarePolsNum,
                                            'evaluationParams': evaluationParams,
                                            'evaluationLog': evaluationLog                                        
                                        }
                                    
    # Compute MAP and MAR
    AP = 0
    prec = []
    rec = []
    confs = []
    if evaluationParams['CONFIDENCES']:
        AP, prec, rec, confs = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt, arrGlobalCareMask)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
    
    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean, 'AP': AP  }

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics}
    
    
    return resDict, prec, rec, confs



if __name__=='__main__':
    pr_curves_folder = './pr_curves/'
    p = {
        'g': './gt_ic15/gt_ic15.zip',
        # 'g': './gt_ic15/gt_ic15_rect.zip',
        # 'g': './gt_ic15/gt_ic15_rect2.zip',
        # 's': './res_nms_ic15/res_nms_ic15_psenet_015_craft_015_weighted_charnet_015.zip'
        # 's': './res_nms_ic15/res_nms_ic15_psenet_93_craft_85_charnet_95.zip'
        # 's': './res_craft_ic15/res_craft_ic15_85_2.zip'
        # 's': './res_craft_ic15/res_craft_ic15_015_weighted.zip'
        # 's': './res_craft_ic15/res_craft_ic15_015_weighted_rect.zip',
        # 's': './res_craft_ic15/res_craft_ic15_015_weighted_rect2.zip',
        # 's': './res_psenet_ic15/res_psenet_ic15_93.zip'
        's': './res_psenet_ic15/res_psenet_ic15_015.zip'
        # 's': './res_psenet_ic15/res_psenet_ic15_015_rect.zip',
        # 's': './res_psenet_ic15/res_psenet_ic15_015_rect2.zip',
        # 's': './res_charnet_ic15/res_charnet_ic15_95.zip'
        # 's': './res_charnet_ic15/res_charnet_ic15_015.zip'
        # 's': './res_charnet_ic15/res_charnet_ic15_015_rect.zip',
        # 's': './res_charnet_ic15/res_charnet_ic15_015_rect2.zip',
        # 's': './res_text_alfa_polygon_ic15/res_text_alfa_polygon_ic15_psenet_015_craft_015_weighted_charnet_015.zip'
        # 's': './res_text_alfa_bbox_ic15/res_text_alfa_bbox_ic15_psenet_015_craft_015_weighted_charnet_015_rect.zip'
        # 's': './res_text_alfa_bbox_ic15/res_text_alfa_bbox_ic15_psenet_015_craft_015_weighted_charnet_015_rect2.zip'
    }
    evalParams = polygon_evaluation_params()
    # evalParams = box_with_angle_evaluation_params()
    result, _, _, _ = rrc_evaluation_funcs.main_evaluation(p, evalParams, validate_data, evaluate_method)

