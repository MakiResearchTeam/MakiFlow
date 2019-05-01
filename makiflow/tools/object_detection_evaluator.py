from __future__ import absolute_import
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from makiflow.ssd.ssd_utils import jaccard_index


class ODEvaluator:
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(num_threads)
        
        
    def mean_average_precision(self, detected_bboxes, gt_bboxes, num_classes, iou_trashhold=0.5):
        """
        Function for calculating mAP over all object categories.
        
        Parameters
        ----------
        detected_boxes : list
            Contains following values for each bouding box: [image_name, class, confidence, [x1, y1, x2, y2]]. 
            Namely `detected_boxes` is a list of lists contain aforementiond values.
        gt_bboxes : list
            Contains following values for each bouding box: [image_name, class, [x1, y1, x2, y2]]. 
            Namely `gt_bboxes` is a list of lists contain aforementiond values.
        
        Returns
        -------
        float
            Mean average precision value.
        """
        # Create container for the calculated average precisions by threads.
        # Look of the items of the container: [average precision, recall, precision]
        aps = []
        for i in range(num_classes):
            detected_boxes_c = [[bbox[0], bbox[2], bbox[3]] for bbox in detected_bboxes if bbox[1] == i]
            gt_bboxes_c = [[bbox[0], bbox[2]] for bbox in gt_bboxes if bbox[1] == i]
            aps.append(ODEvaluator.average_precision( 
                                 detected_bboxes=detected_boxes_c, 
                                 gt_bboxes=gt_bboxes_c,
                                 iou_trashhold=iou_trashhold))

        av_pres = [e[0] for e in aps]
        av_pres = np.array(av_pres)
        mean_average_precision = np.mean(av_pres)
        return [mean_average_precision, aps]
                                
        
        
    
    @staticmethod
    def average_precision(detected_bboxes, gt_bboxes, iou_trashhold=0.5):
        """
        Function for calculating average precision for a particular object category.
        
        Parameters
        ----------
        detected_boxes : list
            Contains following values for each bouding box: [image_name, confidence, [x1, y1, x2, y2]]. 
            Namely `detected_boxes` is a list of lists contain aforementiond values.
        gt_bboxes : list
            Contains following values for each bouding box: [image_name, [x1, y1, x2, y2]]. 
            Namely `gt_bboxes` is a list of lists contain aforementiond values.
        
        Returns
        -------
        list
            [average precision value, recall values, precision values]
        """
        # Sort all the detections in descending order
        detections = sorted(detected_bboxes, key=lambda det: det[1], reverse=True)
        # True Positives
        TP = np.zeros(len(detections))
        # False Negatives
        FP = np.zeros(len(detections))
        # Ground truth boxes have been already seen
        gt_seen = np.zeros(len(gt_bboxes))
        
        for i in range(len(detections)):
            box = np.array([detections[i][2]])
            for j in range(len(gt_bboxes)):
                gt_box = np.array([gt_bboxes[j][1]])
                # Check if detected box and gt box are from the same image
                if detections[i][0] != gt_bboxes[j][0]:
                    continue
                    
                if jaccard_index(box, gt_box)[0] >= iou_trashhold:
                    if gt_seen[j] == 0:
                        TP[i] = 1
                        gt_seen[j] = 1
                    else:
                        FP[i] = 1
                else:
                    FP[i] = 1
                    
        # Compute precision and recall
        acc_TP = np.cumsum(TP)
        acc_FP = np.cumsum(FP)
        recall = acc_TP / len(gt_bboxes)
        precision = np.divide(acc_TP, (acc_TP + acc_FP))
        
        # COMPUTE AVERAGE PRECISION
        # This code is taken from https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
        def CalculateAveragePrecision(rec, prec):
            mrec = []
            mrec.append(0)
            [mrec.append(e) for e in rec]
            mrec.append(1)
            mpre = []
            mpre.append(0)
            [mpre.append(e) for e in prec]
            mpre.append(0)
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            ii = []
            for i in range(len(mrec) - 1):
                if mrec[1:][i] != mrec[0:-1][i]:
                    ii.append(i + 1)
            ap = 0
            for i in ii:
                ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
            return ap

        return [CalculateAveragePrecision(recall, precision), recall, precision]
    
    
            
            
                
                
        