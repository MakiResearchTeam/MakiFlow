from __future__ import absolute_import
import numpy as np
from makiflow.metrics.utils import one_hot

EPSILON = 1e-6


def dice_coeff(y_true, y_pred, depth, flatten=False):
    """
    Computes Dice Coefficient for given labels and predictions.
    Parameters
    ----------
    y_true : array like
        Sparse labels. Can whether a list or numpy array.
    y_pred : np.ndarray
        Predictions produced by softmax. Note that they are not sparse.
    depth : int
        Depth of y_pred tensor, i.e. number of classes.
    flatten : bool
        Set to true if labels and predictions are grouped into batches or
        have any other special shape.
    Returns
    -------
    float
        Computed Dice Coefficient.
    """
    if flatten:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1, depth)

    one_hot_labels = one_hot(y_true, depth)
    intersection = np.sum(one_hot_labels * y_pred)
    union_ish = np.sum(one_hot_labels) + np.sum(y_pred)
    # NOTE: np.sum(y_pred) = len(y_pred)
    # Add `EPSILON` for smoothing
    return (2 * intersection + EPSILON) / (union_ish + EPSILON)

