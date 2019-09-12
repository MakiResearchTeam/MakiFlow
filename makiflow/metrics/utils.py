import numpy as np


def one_hot(sparse_labels, depth):
    """
    Creates one-hot encoding for sparse labels. Note that
    this method supports only 1-dimensional labels (i.e. labels grouped in batches
    or in any other order will cause unpredictable result, you have to flatten your
    labels first)
    Parameters
    ----------
    sparse_labels : array like
        Labels to encode.
    depth : int
        Num classes.
    Returns
    -------
    np.ndarray
        One-hot encoded labels.
    """
    N = len(sparse_labels)
    one_hotted = np.zeros((N, depth), dtype=np.uint8)
    one_hotted[np.arange(N), sparse_labels] = 1
    return one_hotted

