from __future__ import absolute_import
from makiflow.metrics.od_utils import jaccard_index
import numpy as np


def prepare_data_wh(true_boxes, true_labels, dboxes_wh, iou_threshold=0.5):
    """
    Converts training data to appropriate format for the training.

    Parameters
    ----------
    true_boxes : ndarray
        Contains true bboxes for one image. Shape is [num_boxes, 4].
    true_labels : ndarray
        Contains labels for `true_boxes`. Shape is [num_boxes]
    dboxes_wh : array like
        Default boxes array taken from the SSD. Shape is [num_predictions, 4].
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
    offsets = np.zeros((num_predictions, 4), dtype=np.float32)
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
        offsets[jaccard_boolean] = _g_hat(true_boxes[i], dboxes_wh[jaccard_boolean])

    return mask.astype(np.float32), labels.astype(np.int32), offsets


def _g_hat(gbox, bboxes_wh):
    g_hat_cx = ((gbox[0] - bboxes_wh[:, 0]) / bboxes_wh[:, 2]).reshape(1, 1)
    g_hat_cy = ((gbox[1] - bboxes_wh[:, 1]) / bboxes_wh[:, 3]).reshape(-1, 1)
    g_hat_w = np.log(gbox[2] / bboxes_wh[:, 2]).reshape(-1, 1)
    g_hat_h = np.log(gbox[:, :, 3] / bboxes_wh[:, 3]).reshape(-1, 1)
    g_hat = np.concatenate([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
    return g_hat
