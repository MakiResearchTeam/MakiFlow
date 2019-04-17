import tensorflow as tf
import numpy as np

def jaccard_index(box_a, box_b):
    """
    Calculates Jaccard Index for two bounding boxes.
    Argumnets:
    box_a - [x_center, y_center, width, height] of the box A.
    box_a - [x_center, y_center, width, height] of the box B.

    Returns number represents Jaccard Index of the aforementioned bboxes.
    Example: 0.7


    Code has taken from:
    https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection/blob/master/data_prep.py

    """
    box_a = [box_a[0] - box_a[2] / 2,  # upper left x
             box_a[1] - box_a[3] / 2,  # upper left y
             box_a[0] + box_a[2] / 2,  # bottom right x
             box_a[1] + box_a[3] / 2]  # bottom right y

    box_b = [box_b[0] - box_b[2] / 2,  # upper left x
             box_b[1] - box_b[3] / 2,  # upper left y
             box_b[0] + box_b[2] / 2,  # bottom right x
             box_b[1] + box_b[3] / 2]  # bottom right y
             
    # Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
    # http://math.stackexchange.com/a/99576
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou
             
             
             
def prepare_data(image_info, dboxes, iou_trashhold=0.5):
    """ Converts training data to appropriate format for the training.
    Arguments:
    image_ingo - dictionary contains info about ground truth bounding boxes and each class assigned to them.
    Example: { 'bboxes': [
                        [x1, y1, w1, h1],
                        [x2, y2, w2, h2],
                        [x3, y3, w3, h3]
                        ],
                'classes': [class1, class2, class3]
                }
    dboxes - default boxes array has taken from the SSD.
    iou_trashhold - Jaccard index dbox must exceed to be marked as positive.
    """
    loc_mask = np.zeros(len(dboxes))
    labels = np.zeros(len(dboxes))
    i = 0
    for gbox in image_info['bboxes']:
        j = 0
        for dbox in dboxes:
            if jaccard_index(gbox, dbox) > iou_trashhold:
                mask[j] = 1
                labels[j] = image_info['classes'][i] # set the of current gbox
            j += 1
        i += 1
    
    return {'loc_mask': loc_mask,
            'labels': labels }

             