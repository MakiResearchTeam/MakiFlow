# For resizing images and bboxes
from copy import copy

# For drawing predicted bounding boxes
import cv2
import numpy as np
from tqdm import tqdm


def jaccard_index(boxes_a, boxes_b):
    """
    Calculates Jaccard Index for pairs of bounding boxes.
    :param boxes_a - list of "first" bboxes. Example:
    [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ]
    :param boxes_b - list of "second" bboxes. Example:
    [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ]
    :return Returns a list of Jaccard indeces for each bbox pair.
    Example: [jaccard_index1, jaccard_index2]

    """

    # box_a = [box_a[0] - box_a[2] / 2,  # upper left x
    #         box_a[1] - box_a[3] / 2,  # upper left y
    #         box_a[0] + box_a[2] / 2,  # bottom right x
    #         box_a[1] + box_a[3] / 2]  # bottom right y

    # box_b = [box_b[0] - box_b[2] / 2,  # upper left x
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

    x_overlap_prev = np_min(boxes_a[:, 2], boxes_b[:, 2]) - np_max(boxes_a[:, 0], boxes_b[:, 0])
    x_overlap_prev = np.vstack([zeros, x_overlap_prev])
    x_overlap = np.max(x_overlap_prev, axis=0)

    y_overlap_prev = np_min(boxes_a[:, 3], boxes_b[:, 3]) - np_max(boxes_a[:, 1], boxes_b[:, 1])
    y_overlap_prev = np.vstack([zeros, y_overlap_prev])
    y_overlap = np.max(y_overlap_prev, axis=0)

    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_box_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou


def prepare_data(image_info, dboxes, iou_trashhold=0.5):
    """
    Converts training data to appropriate format for the training.
    
    Parameters
    ----------
    image_info : dictionary
        Contains info about ground truth bounding boxes and each class assigned to them.
        Example: { 'bboxes': [
                            [x1, y1, x2, y2],
                            [x1, y1, x2, y2],
                            [x1, y1, x2, y2]
                            ],
                    'classes': [class1, class2, class3]
                    }, 
        where class1, class2, class3 are ints.
    dboxes : array like
        Default boxes array taken from the SSD.
    iou_trashhold : float
        Jaccard index dbox must exceed to be marked as positive.
         
    Returns
    -------
    dictionary
        Contains `loc_mask` masks for localization loss, (sparse) `labels` vector with class labels and
        `locs` vector contain differences in coordinates between ground truth boxes and default boxes which
        will be used for the calculation of the localization loss.
        Example: {  'loc_mask': ...,
                    'labels'  : ...,
                    'gt_locs' : ...  }
    """
    num_predictions = len(dboxes)
    loc_mask = np.array([0] * num_predictions)
    labels = np.zeros(num_predictions)
    # Difference between ground true box and default box. Need it for the later loss calculation.
    locs = np.zeros((num_predictions, 4))
    i = 0
    for gbox in image_info['bboxes']:
        j = 0
        gbox_stack = np.vstack([gbox] * num_predictions)
        jaccard_indexes = jaccard_index(gbox_stack, dboxes)
        # Use logic "or"
        jaccard_boolean = jaccard_indexes > iou_trashhold
        loc_mask = jaccard_boolean | loc_mask
        labels[jaccard_boolean] = image_info['classes'][i]
        locs[jaccard_boolean] = gbox - dboxes[jaccard_boolean]
        i += 1

    return {'loc_mask': loc_mask,
            'labels': labels,
            'gt_locs': locs}


def prepare_data_v2(true_boxes, true_labels, dboxes, iou_threshold=0.5):
    """
    Converts training data to appropriate format for the training.

    Parameters
    ----------
    true_boxes : ndarray
        Contains true bboxes for one image.
    true_labels : ndarray
        Contains labels for `true_boxes`.
    dboxes : array like
        Default boxes array taken from the SSD.
    iou_threshold : float
        Jaccard index dbox must exceed to be marked as positive.

    Returns
    -------
    labels : ndarray
        Ndarray of labels (int32).
    locs : ndarray
        Ndarray of binary localization masks (float32).
    loc_mask : ndarray
        Ndarray of localization masks (float32).
    """
    num_predictions = len(dboxes)
    loc_mask = np.zeros(num_predictions, dtype=np.int8)
    labels = np.zeros(num_predictions, dtype=np.int8)
    # Difference between ground true box and default box. Need it for the later loss calculation.
    locs = np.zeros((num_predictions, 4), dtype=np.float32)
    for i in range(len(true_boxes)):
        true_box_stack = np.vstack([true_boxes[i]] * num_predictions)
        jaccard_indexes = jaccard_index(true_box_stack, dboxes)
        # Choose positive dboxes
        jaccard_boolean = jaccard_indexes > iou_threshold
        # Mark positive dboxes
        loc_mask = jaccard_boolean | loc_mask
        # Mark positive dboxes with labels
        labels[jaccard_boolean] = true_labels[i]
        # Calculate localizations for positive dboxes
        locs[jaccard_boolean] = true_boxes[i] - dboxes[jaccard_boolean]

    return labels.astype(np.int32), locs, loc_mask.astype(np.float32)


def draw_bounding_boxes(image, bboxes_with_classes):
    """
    Draw bounding boxes on the image.
    
    Parameters
    ----------
    image : numpy ndarray
        Image the bboxes will be drawn on. It is numpy array with shape [image_w, image_h, color_channels]
    bboxes_with_classes : python dictionary
        Dictionary with bboxes and predicted classes. Example:
        {'bboxes':  [
                    [x1, y1, x2, y2],
                    [x1, y1, x2, y2]
                    ]
        'classes': ['class1', 'class2']
        }
    
    Returns
    -------
    numpy ndarray
        Image with drawn bounding boxes on it.
    """
    prediction_num = len(bboxes_with_classes['bboxes'])
    image_copy = copy(image)
    for i in range(prediction_num):
        box_coords = bboxes_with_classes['bboxes'][i]
        box_coords = [int(coord) for coord in box_coords]

        image_copy = cv2.rectangle(image_copy, tuple(box_coords[:2]), tuple(box_coords[2:]), (0, 255, 0))
        label_str = bboxes_with_classes['classes'][i]
        image_copy = cv2.putText(image_copy, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 255, 0), 1,
                                 cv2.LINE_AA)
    return image_copy


def resize_images_and_bboxes(image_array, bboxes_array, new_size):
    """
    Resizes images accordingly with new_size.
    image_array - list of images: [image1, image2, image3, ...]
    :param bboxes_array - list of bboxes to be scaled with images. Example:
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
    :param new_size - tuple of numbers represent new size: (new_width, new_height)
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
            new_bboxes_array[i][j][0] = round(bbox[0] * width_ratio)
            new_bboxes_array[i][j][2] = round(bbox[2] * width_ratio)
            new_bboxes_array[i][j][1] = round(bbox[1] * height_ratio)
            new_bboxes_array[i][j][3] = round(bbox[3] * height_ratio)
            j += 1
        i += 1

    return new_image_array, new_bboxes_array


def resize_images_and_bboxes_v2(images, bboxes, new_size):
    """
    Resizes images accordingly to `new_size`.
    Parameters
    ----------
    images : list
        List of images (ndarrays).
    bboxes : list
        List of images (ndarrays).
    new_size : tuple
        (new_width, new_height).

    Returns
    -------
    new_images : list
        Resized images. Same structure as `images`.
    new_bboxes : list
        Resized bboxes. Same structure as `true_boxes`.
    """
    new_images = []
    new_bboxes = []

    for image, bboxes in tqdm(zip(images, bboxes)):
        image_dims = image.shape[:2]
        width_ratio = new_size[0] / image_dims[1]
        height_ratio = new_size[1] / image_dims[0]
        new_images += [cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)]
        # Creating new bbox
        n_bboxes = np.copy(bboxes)
        n_bboxes[:, 0] *= width_ratio
        n_bboxes[:, 2] *= width_ratio
        n_bboxes[:, 1] *= height_ratio
        n_bboxes[:, 3] *= height_ratio
        new_bboxes += [n_bboxes]

    return new_images, new_bboxes


def nms(pred_bboxes, pred_confs, conf_trashhold=0.4, iou_trashhold=0.1, background_class=0):
    """
    Performs Non-Maximum Suppression on predicted bboxes.
    :param pred_bboxes - list of predicted bboxes. Numpy array of shape [num_predictions, 4].
    :param pred_confs - list of predicted confidences. Numpy array of shape [num_predictions, num_classes].
    
    conf_trashhold : float
        All the predictions with the confidence less than `conf_trashhold` will be treated
        as negatives.
    iou_trashhold : float
        Used for performing Non-Maximum Supression. NMS pickes the most confident detected
        bounding box and deletes all the bounding boxes have IOU(Jaccard Index) more
        than `iou_trashhold`. LESSER - LESS BBOXES LAST, MORE - MORE BBOXES LAST.
        
    :param background_class - index of the background class.
    :return Returns final predicted bboxes and confidences
    """
    # TODO: backgroung_class is never used

    pred_conf_values = np.max(pred_confs, axis=1)

    # Take predicted boxes with confidence higher than conf_trash_hold
    filtered_pred_boxes = pred_bboxes[pred_conf_values > conf_trashhold]
    filtered_pred_confs = pred_confs[pred_conf_values > conf_trashhold]

    # Get classes in order get rid of background class
    pred_conf_classes = np.argmax(filtered_pred_confs, axis=1)

    # Get rid of background class boxes
    back_ground_class_mask = pred_conf_classes != 0
    filtered_pred_boxes = filtered_pred_boxes[back_ground_class_mask]
    filtered_pred_confs = filtered_pred_confs[back_ground_class_mask]

    # Create array with indexes of bboxes for easier navigation later
    indexes = np.arange(filtered_pred_confs.shape[0])

    pred_conf_classes = np.argmax(filtered_pred_confs, axis=1)
    pred_conf_values = np.max(filtered_pred_confs, axis=1)

    usage_mask = pred_conf_values > conf_trashhold
    final_boxes = []
    final_conf_classes = []
    final_conf_values = []
    while True:
        # Step 1: Choose the most confident box if possible
        bboxes = filtered_pred_boxes[usage_mask]
        conf_values = pred_conf_values[usage_mask]
        conf_classes = pred_conf_classes[usage_mask]
        unused_indexes = indexes[usage_mask]
        if bboxes.shape[0] == 0:
            break

        id_most_confident = np.argmax(conf_values)
        most_confident_box = bboxes[id_most_confident]

        # Step 2: Mark the box as used in the mask, add to final predictions
        usage_mask[unused_indexes[id_most_confident]] = False
        final_boxes.append(most_confident_box)
        final_conf_classes.append(conf_classes[id_most_confident])
        final_conf_values.append(conf_values[id_most_confident])

        # Step 3: Calculate Jaccard Index for the boxes with the same class

        # Pick boxes with the same class
        boxes_same_class = bboxes[conf_classes == conf_classes[id_most_confident]]
        # Save indexes of these boxes
        indexes_same_class = unused_indexes[conf_classes == conf_classes[id_most_confident]]
        # Stack current boxes for the Jaccard Index calucaltion
        most_confident_box_stack = np.vstack([most_confident_box] * len(boxes_same_class))
        ious = jaccard_index(most_confident_box_stack, boxes_same_class)

        # Step 4: Mark boxes as used which iou is greater than 0.1
        boxes_to_mark = indexes_same_class[ious > iou_trashhold]
        for box in boxes_to_mark:
            usage_mask[box] = False

    return final_boxes, final_conf_classes, final_conf_values
