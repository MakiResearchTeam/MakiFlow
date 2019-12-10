from __future__ import absolute_import
from makiflow.metrics.od_utils import jaccard_index
import numpy as np
from tqdm import tqdm


def prepare_data_v2(true_boxes, true_labels, dboxes_wh, iou_threshold=0.5):
    """
    Converts training data to appropriate format for the training.

    Parameters
    ----------
    true_boxes : ndarray
        Contains true bboxes for one image.
    true_labels : ndarray
        Contains labels for `true_boxes`.
    dboxes_wh : array like
        Default boxes array taken from the SSD.
    iou_threshold : float
        Jaccard index dbox must exceed to be marked as positive.

    Returns
    -------
    loc_mask : ndarray
        Mask vector for positive in-image samples (float32).
    labels : ndarray
        Ndarray of labels (int32).
    locs : ndarray
        Ndarray of binary localization masks (float32).

    """
    num_predictions = len(dboxes_wh)
    mask = np.zeros(num_predictions, dtype=np.int32)
    labels = np.zeros(num_predictions, dtype=np.int32)
    # Difference between ground true box and default box. Need it for the later loss calculation.
    locs = np.zeros((num_predictions, 4), dtype=np.float32)
    for i in range(len(true_boxes)):
        true_box_stack = np.vstack([true_boxes[i]] * num_predictions)
        jaccard_indexes = jaccard_index(true_box_stack, dboxes_wh)
        # Choose positive dboxes
        jaccard_boolean = jaccard_indexes > iou_threshold
        # Mark positive dboxes
        mask = jaccard_boolean | mask
        # Mark positive dboxes with labels
        labels[jaccard_boolean] = true_labels[i]
        # Calculate localizations for positive dboxes
        locs[jaccard_boolean] = true_boxes[i] - dboxes_wh[jaccard_boolean]

    return mask.astype(np.float32), labels.astype(np.int32), locs
