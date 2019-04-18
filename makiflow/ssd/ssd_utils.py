import tensorflow as tf
import numpy as np
# For drawing predicted bounding boxes
import cv2
# For resizing images and bboxes
from copy import copy
from tqdm import tqdm

def jaccard_index(box_a, box_b):
    """
    Calculates Jaccard Index for pairs of bounding boxes.
    Argumnets:
    box_a - list of "first" bboxes. Example:
    [
        [x1, y1, 2, y2],
        [x1, y1, 2, y2]
    ]
    box_a - list of "second" bboxes. Example:
    [
        [x1, y1, 2, y2],
        [x1, y1, 2, y2]
    ]
    Returns a list of Jaccard indeces for each bbox pair.
    Example: [jaccard_index1, jaccard_index2]

    """
    #box_a = [box_a[0] - box_a[2] / 2,  # upper left x
    #         box_a[1] - box_a[3] / 2,  # upper left y
    #         box_a[0] + box_a[2] / 2,  # bottom right x
    #         box_a[1] + box_a[3] / 2]  # bottom right y

    #box_b = [box_b[0] - box_b[2] / 2,  # upper left x
    #         box_b[1] - box_b[3] / 2,  # upper left y
    #         box_b[0] + box_b[2] / 2,  # bottom right x
    #         box_b[1] + box_b[3] / 2]  # bottom right y
             
    # Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
    # http://math.stackexchange.com/a/99576
    def np_min(a, b):
        ab = np.vstack([a, b])
        return np.min(ab, axis=0)

    def np_max(a, b):
        ab = np.vstack([a, b])
        return np.max(ab, axis=0)
    
    zeros = np.zeros(len(boxes_a))
    
    x_overlap_prev = np_min(boxes_a[:,2], boxes_b[:,2]) - np_max(boxes_a[:,0], boxes_b[:,0])
    x_overlap_prev = np.vstack([zeros, x_overlap_prev])
    x_overlap = np.max(x_overlap_prev, axis=0)
    
    y_overlap_prev = np_min(boxes_a[:,3], boxes_b[:,3]) - np_max(boxes_a[:,1], boxes_b[:,1])
    y_overlap_prev = np.vstack([zeros, y_overlap_prev])
    y_overlap = np.max(y_overlap_prev, axis=0)
    
    intersection = x_overlap * y_overlap
    
    # Calculate union
    area_box_a = (boxes_a[:,2] - boxes_a[:,0]) * (boxes_a[:,3] - boxes_a[:,1])
    area_box_b = (boxes_b[:,2] - boxes_b[:,0]) * (boxes_b[:,3] - boxes_b[:,1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou
             
             
def prepare_data(image_info, dboxes, iou_trashhold=0.5):
    """ Converts training data to appropriate format for the training.
    Arguments:
    image_ingo - dictionary contains info about ground truth bounding boxes and each class assigned to them.
    Example: { 'bboxes': [
                        [x1, y1, x2, y2],
                        [x1, y1, x2, y2],
                        [x1, y1, x2, y2]
                        ],
                'classes': [class1, class2, class3]
                }
    dboxes - default boxes array has taken from the SSD.
    iou_trashhold - Jaccard index dbox must exceed to be marked as positive.
    """
    num_predictions = len(dboxes)
    loc_mask = np.zeros(num_predictions)
    labels = np.zeros(num_predictions)
    # Difference between ground true box and default box. Need it for the later loss calculation.
    locs = np.zeros((num_predictions, 4))
    i = 0
    for gbox in image_info['bboxes']:
        j = 0
        gbox_stack = np.vstack([gbox]*num_predictions)
        jaccard_indeces = jaccard_index(gbox_stack, dboxes)
        for dbox in dboxes:
            if jaccard_index(gbox, dbox) > iou_trashhold:
                loc_mask[j] = 1
                labels[j] = image_info['classes'][i] # set the class of current gbox
                locs[j] = gbox - dbox
            j += 1
        i += 1
    
    return {'loc_mask': loc_mask,
            'labels': labels,
            'gt_locs': locs}


def draw_bounding_boxes(image, bboxes_with_classes):
    """ Draw bounding boxes on the image.
    image - image the bboxes will be drawn on.
    bboxes_with_classes - dictionary with bboxes and predicted classes. Example:
        {'bboxes':  [
                    [x1, y1, x2, y2],
                    [x1, y1, x2, y2]
                    ]
        'classes': ['class1', 'class2']
        }
        
    Returns image with drawn bounding box on it.
    """
    prediction_num = len(bboxes_with_classes['bboxes'])
    for i in range(prediction_num):
        box_coords = bboxes_with_classes['bboxes'][i]
        box_coords = [int(coord) for coord in box_coords]
        
        image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:]), (0,255,0))
        label_str = bboxes_with_classes['classes'][i]
        image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return image


def resize_images_and_bboxes(image_array, bboxes_array, new_size):
    """ Resizes images accordingly with new_size.
    image_array - list of images: [image1, image2, image3, ...]
    bboxes_array - list of bboxes to be scaled with images. Example:
        [
            # bboxes for the first image
            [
                [x1, y1, x2, y2],
                [x1, y1, x2, y2]
            ],
            # bboxes for the second image
            [
                [x1, y1, x2, y2],
                [x1, y1, x2, y2]
            ]
        ]
    new_size - tuple of numbers represent new size: (new_width, new_height)
    """
    new_image_array = copy(image_array)
    new_bboxes_array = copy(bboxes_array)
    # For navigation in new_image_array and new_bboxes_array
    i = 0
    for image, bboxes in tqdm(zip(image_array, bboxes_array)):
        image_dims = image.shape[:2]
        width_ratio = new_size[0] / image_dims[1]
        height_ratio = new_size[1] / image_dims[0]
        new_image_array[i] = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        # For navigation in new_bboxes_array
        j = 0
        for bbox in bboxes:
            new_bboxes_array[i][j][0] = round(bbox[0]*width_ratio)
            new_bboxes_array[i][j][2] = round(bbox[2]*width_ratio)
            new_bboxes_array[i][j][1] = round(bbox[1]*height_ratio)
            new_bboxes_array[i][j][3] = round(bbox[3]*height_ratio)
            j += 1
        i += 1
        
    return new_image_array, new_bboxes_array


def nms(pred_bboxes, pred_confs, conf_trash_hold):
    """ Performs Non-Maximum Supression on predicted bboxes.
    Arguments:
    pred_bboxes - list of predicted bboxes.
    pred_confs - list of predicted confidences.
    conf_trash_hold - used for filtering bboxes
    
    Returns final predicted bboxes and confidences
    """
    pass