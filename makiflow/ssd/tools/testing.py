from __future__ import absolute_import
from makiflow.ssd.utils import nms
from makiflow.tool.object_detection_evaluator import ODEvaluator
from tqdm import tqdm

class SSDTester:
    @staticmethod
    def mean_average_precision(ssd, images, annotation_dict):
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
            filtered_boxes.append(ssd_utils.nms(bboxes_list[i], confidences_list[i], conf_trashhold=0.5, iou_trashhold=0.5))
            
            
        # Create list of marked detected boxes
        detected_boxes = []
        for i in range(confidences_list.shape[0]):
            image_name = annotation_dict[i]['filename']
            # No detections
            if len(filtered_boxes[i][0]) == 0:
                continue

            for j in range(len(filtered_boxes[i][0])):
                bb_coords = filtered_boxes[i][0][j]
                bb_class = filtered_boxes[i][1][j]-1
                bb_confidence = filtered_boxes[i][2][j]
                detected_boxes.append([image_name, bb_class, bb_confidence, bb_coords])
                
        # Create list of marked ground truth boxes
        gt_boxes = []
        for i in range(confidences_list.shape[0]):
            image_name = annotation_dict[i]['filename']
            for gt_bb in annotation_dict[i]['objects']:
                bb_coords = gt_bb['box']
                bb_class = class_inds[gt_bb['name']]-1
                gt_boxes.append([image_name, bb_class, bb_coords])
                
        return evaluator.mean_average_precision(detected_boxes, gt_boxes, num_classes=ssd.num_classes)

        