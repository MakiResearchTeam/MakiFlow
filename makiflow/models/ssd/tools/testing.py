# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import
from makiflow.models.ssd.ssd_utils import nms
from makiflow.tools.object_detection_evaluator import ODEvaluator
from tqdm import tqdm
import numpy as np

class SSDTester:
    def __init__(self):
        pass
    
    
    def prepare_ground_truth_labels(self, annotation_dict, class_name_to_num):
        """
        Prepares ground truth data for later testing.
        
        Parameters
        ----------
        annotation_dict : dictionary
            Dictionary with annotations. You take it from the xml or json parsers. You
            need pass in the dictionary contains labels only for images gonna be processed
            by the SSD.
        class_name_to_num : dictionary
            Maps class names with their indices. WARNING! INDICES MUST START COUNTING FROM 1! Example:
            {
                'cat' : 1,
                'dog' : 2,
                'car' : 3
            }
        """
        self.annotation_dict = annotation_dict
        self.class_name_to_num = class_name_to_num
        # Create list of marked ground truth boxes
        self.gt_boxes = []
        for i in range(len(annotation_dict)):
            image_name = annotation_dict[i]['filename']
            for gt_bb in annotation_dict[i]['objects']:
                bb_coords = gt_bb['box']
                bb_class = class_name_to_num[gt_bb['name']]-1
                self.gt_boxes.append([image_name, bb_class, bb_coords])
        print('Number of ground truth detections:', len(self.gt_boxes))
                
                
    def mean_average_precision(self, ssd, images, conf_trashhold=0.5, iou_trashhold=0.5):
        """
        Computes mean average precision given predictions.
        
        Parameters
        ----------
        ssd : MakiFlow SSDModel
            SSDModel will be used for making predictions.
        images : list
            List of images to process by the SSD.
        annotation_dict : dictionary
            Dictionary with annotations. You take it from the xml or json parsers. You
            need pass in the dictionary contains labels only for images gonna be processed
            by the SSD. Annotation indeces must match images' indeces.
        conf_trashhold : float
            All the predictions with the confidence less than `conf_trashhold` will be treated
            as negatives.
        iou_trashhold : float
            Used for performing Non-Maximum Supression. NMS pickes the most confident detected
            bounding box and deletes all the bounding boxes have IOU(Jaccard Index) more
            than `iou_trashhold`. LESSER - LESS BBOXES LAST, MORE - MORE BBOXES LAST.
        
        Returns
        -------
        list
            Contains two values: mAP and list AVs for each class.
        """
            
        # Process all images
        batch_size = ssd.input_shape[0]
        num_batches = len(images) // batch_size
        confidences_list = []
        bboxes_list = []
        print('Processing images...')
        for i in tqdm(range(num_batches)):
            image_batch = images[i*batch_size: (i+1)*batch_size]
            confidences, bboxes = ssd.predict(image_batch)
            confidences_list += [confidences]
            bboxes_list += [bboxes]
            
        confidences_list = np.vstack(confidences_list)
        bboxes_list = np.vstack(bboxes_list)
            
            
        # Filter predictions with Non-Maximum Supression
        print('Filtering detections with NMS...')
        filtered_boxes = []
        for i in tqdm(range(confidences_list.shape[0])):
            filtered_boxes.append(nms(bboxes_list[i],
                                                confidences_list[i],
                                                conf_trashhold=conf_trashhold,
                                                iou_trashhold=iou_trashhold))
                
        # Create list of marked detected boxes
        detected_boxes = []
        for i in range(confidences_list.shape[0]):
            image_name = self.annotation_dict[i]['filename']
            # No detections
            if len(filtered_boxes[i][0]) == 0:
                continue

            for j in range(len(filtered_boxes[i][0])):
                bb_coords = filtered_boxes[i][0][j]
                bb_class = filtered_boxes[i][1][j]-1
                bb_confidence = filtered_boxes[i][2][j]
                detected_boxes.append([image_name, bb_class, bb_confidence, bb_coords])
        print('Number of the SSD detections:', len(detected_boxes))
        return ODEvaluator.mean_average_precision(detected_boxes, self.gt_boxes, num_classes=ssd.num_classes-1)

        