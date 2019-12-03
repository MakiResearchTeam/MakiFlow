import numpy as np


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
    :return Returns a list of Jaccard indices for each bbox pair.
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


def nms(pred_bboxes, pred_confs, conf_threshold=0.4, iou_threshold=0.1, background_class=0):
    """
    Performs Non-Maximum Suppression on predicted bboxes.

    Parameters
    ----------
    pred_bboxes : list
        List of predicted bboxes. Numpy array of shape [num_predictions, 4].
    pred_confs : list
        List of predicted confidences. Numpy array of shape [num_predictions, num_classes].
    conf_threshold : float
        All the predictions with the confidence less than `conf_threshold` will be treated
        as negatives.
    iou_threshold : float
        Used for performing Non-Maximum Suppression. NMS picks the most confident detected
        bounding box and deletes all the bounding boxes that have IOU(Jaccard Index) more
        than `iou_threshold`. LESSER - LESS BBOXES LAST, MORE - MORE BBOXES LAST.
    background_class : int
        Index of the background class.

    Returns
    -------
    finale : list
        List of ndarrays with bbox coordinates.
    final_conf_classes : list
        List of classes (ints) for the predicted `bboxes`.
    final_conf_values : list
        List of confidences (floats) for the predicted `classes`.
    """

    pred_conf_values = np.max(pred_confs, axis=1)

    # Take predicted boxes with confidence higher than conf_trash_hold
    filtered_pred_boxes = pred_bboxes[pred_conf_values > conf_threshold]
    filtered_pred_confs = pred_confs[pred_conf_values > conf_threshold]

    # Get classes in order get rid of background class
    pred_conf_classes = np.argmax(filtered_pred_confs, axis=1)

    # Get rid of background class boxes
    back_ground_class_mask = pred_conf_classes != background_class
    filtered_pred_boxes = filtered_pred_boxes[back_ground_class_mask]
    filtered_pred_confs = filtered_pred_confs[back_ground_class_mask]

    # Create array with indexes of bboxes for easier navigation later
    indexes = np.arange(filtered_pred_confs.shape[0])

    pred_conf_classes = np.argmax(filtered_pred_confs, axis=1)
    pred_conf_values = np.max(filtered_pred_confs, axis=1)

    usage_mask = pred_conf_values > conf_threshold
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
        # Stack current boxes for the Jaccard Index calculation
        most_confident_box_stack = np.vstack([most_confident_box] * len(boxes_same_class))
        ious = jaccard_index(most_confident_box_stack, boxes_same_class)

        # Step 4: Mark boxes as used which iou is greater than 0.1
        boxes_to_mark = indexes_same_class[ious > iou_threshold]
        for box in boxes_to_mark:
            usage_mask[box] = False

    return final_boxes, final_conf_classes, final_conf_values


def compute_tps(pred_boxes, pred_classes, true_boxes, true_classes, iou_th=0.5):
    """
    Computes True Positives for the given predictions.

    Parameters
    ----------
    pred_boxes : ndarray
        Array of the predicted bounding boxes.
    pred_classes : ndarray
        Array of the predicted classes for `pred_boxes`.
    true_boxes : ndarray
        Array of the true bounding boxes.
    true_classes : ndarray
        Array of the true classes for `true_boxes`.

    Returns
    -------
    ndarray
        Array of shape [len(pred_boxes)] where ith element equals 1 if ith bbox is correct
        and 0 otherwise.

    """
    tps = np.zeros(len(pred_boxes))
    for i, true_box in enumerate(true_boxes):
        ious = jaccard_index(pred_boxes, np.array([true_box] * len(pred_boxes)))
        iou_match = ious > iou_th
        class_match = pred_classes == true_classes[i]
        # Take the only bboxes that match both IoU and class requirements
        tp_indices = iou_match * class_match
        # If there is at least one True value
        if np.max(tp_indices):
            tps[np.argmax(tp_indices)] = 1
    return tps


def clear_filtered_preds(filtered_preds):
    """
    Clears NMS result from empty predictions.
    Parameters
    ----------
    filtered_preds : list
        [(bboxes, classes, confidences)], where
        `bboxes` is a list of ndarrays with bbox coordinates;
        `classes` is a list of classes (ints) for the predicted `bboxes`;
        `confidences` is a list of confidences (floats) for the predicted `classes`.

    Returns
    -------
    cleared_preds : list
        Same as `filtered_preds` but without empty results.
    """
    # Deletes empty predictions
    cleared = []
    for filtered_pred in filtered_preds:
        if len(filtered_pred[0]) != 0:
            cleared += [filtered_pred]
    return cleared


def merge_nms_result(filtered_pred):
    """
    Concatenates list of bboxes into single ndarray.
    Converts lists of classes and confidences into ndarrays.
    It is needed for the later calculations of the mAP.
    Parameters
    ----------
    filtered_pred : list
        [(bboxes, classes, confidences)], where
        `bboxes` is a list of ndarrays with bbox coordinates;
        `classes` is a list of classes (ints) for the predicted `bboxes`;
        `confidences` is a list of confidences (floats) for the predicted `classes`.

    Returns
    -------
    merged_filtered_pred : list
        [(bboxes, classes, confidences)], where
        `bboxes` is a ndarray of ndarrays with bbox coordinates;
        `classes` is a ndarray of classes (ints) for the predicted `bboxes`;
        `confidences` is a ndarray of confidences (floats) for the predicted `classes`.

    """
    boxes = np.vstack(filtered_pred[0])
    classes = np.array(filtered_pred[1])
    confs = np.array(filtered_pred[2])
    return boxes, classes, confs


def parse_dicts(data_dicts, name2class):
    """
    Creates arrays of bboxes and the corresponding classes from the given
    dictionaries in the `data_dicts`.
    `data_dicts` are usually obtained after reading .xml files in PascalVOC dataset.
    Parameters
    ----------
    data_dicts : list
        `data_dict` is a list of dictionaries.
        The dictionaries contain several fields:
        'filename' - name of the file;
        'folder' - name of the folder;
        'size' - size of the picture: (channels, width, height);
        'objects' - list of dictionaries that contain info about the
        objects on the image:
        [{'name' : 'object_name', 'box': [x1, x2, x3, x4]}].
        Actual view:
        data_dicts = [
            {
                'filename': filename,
                'folder': folder name,
                'size': (channels, width, height),
                'objects': [
                    {
                        'name': class name,
                        'box': [x1, x2, x3, x4]
                    }
                ]
            }
        ]
    name2class : dict
        Mapping from the class names to their indices.

    Returns
    -------
    true_boxes : list
        List of ndarrays (concatenated bboxes).
    true_classes : list
        List of ndarrays (arrays of the classes of the `true_boxes`).
    """
    # `data_dict` is a list of dictionaries.
    # The dictionaries contain several fields:
    # 'filename' - name of the file;
    # 'folder' - name of the folder;
    # 'size' - size of the picture: (channels, width, height);
    # 'objects' - list of dictionaries that contain info about the
    # objects on the image:
    # [{'name' : 'object_name', 'box': [x1, x2, x3, x4]}]
    true_boxes = []
    true_classes = []
    for image_dict in data_dicts:
        # Collect the bboxes and stack them together
        image_boxes = []
        image_classes = []
        for object_dict in image_dict['objects']:
            image_boxes += [np.array(object_dict['box'], dtype=np.float32)]
            image_classes += [name2class[object_dict['name']]]
        true_boxes += [np.vstack(image_boxes)]
        true_classes += [np.array(image_classes, dtype=np.int32)]
    return true_boxes, true_classes

