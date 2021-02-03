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
from makiflow.metrics.od_utils import jaccard_index
from makiflow.old.models import bboxes_xy2wh
import numpy as np


def prepare_data_rcnn(gboxes_xy, true_labels, dboxes_wh, dboxes_xy, iou_threshold=0.3):
    """
    Converts training data to appropriate format for the training.

    Parameters
    ----------
    gboxes_xy : ndarray
        Contains true bboxes for one image. Shape is [num_boxes, 4].
    true_labels : ndarray
        Contains labels for `true_boxes`. Shape is [num_boxes]
    dboxes_wh : ndarray
        Default boxes in the WH format taken from the SSD. Shape is [num_predictions, 4].
    dboxes_xy : ndarray
        Default boxes in the XY format taken from the SSD. Shape is [num_predictions, 4].
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
    # `gboxes_xy` are used for IoU calculation.
    # `gboxes_wh` are used for the offsets calculation.
    gboxes_wh = bboxes_xy2wh(gboxes_xy)
    for i in range(len(gboxes_xy)):
        true_box_stack = np.vstack([gboxes_xy[i]] * num_predictions)
        jaccard_indexes = jaccard_index(true_box_stack, dboxes_xy)
        # Choose positive dboxes
        jaccard_boolean = jaccard_indexes > iou_threshold
        # Mark positive dboxes
        mask = jaccard_boolean | mask
        # Mark positive dboxes with labels
        labels[jaccard_boolean] = true_labels[i]
        # Calculate localizations for positive dboxes
        offsets[jaccard_boolean] = _g_hat(gboxes_wh[i], dboxes_wh[jaccard_boolean])

    return mask.astype(np.float32), labels.astype(np.int32), offsets


def _g_hat(gbox, bboxes_wh):
    g_hat_cx = ((gbox[0] - bboxes_wh[:, 0]) / bboxes_wh[:, 2]).reshape(-1, 1)
    g_hat_cy = ((gbox[1] - bboxes_wh[:, 1]) / bboxes_wh[:, 3]).reshape(-1, 1)
    g_hat_w = np.log(gbox[2] / bboxes_wh[:, 2]).reshape(-1, 1)
    g_hat_h = np.log(gbox[3] / bboxes_wh[:, 3]).reshape(-1, 1)
    g_hat = np.concatenate([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
    return g_hat
