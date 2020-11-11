import numpy as np


def make_box(keypoints: np.ndarray):
    """
    Creates a bounding box that encompasses given `keypoints`.
    Parameters
    ----------
    keypoints : ndarray of shape [n, 2]
        Keypoints for which to create the bounding box.

    Returns
    -------
    ndarray of shape [4]
        [x_left, y_up, x_right, y_down]
    """
    assert keypoints.shape[1] == 2, f'keypoints are not 2-dimensional. Received shape={keypoints.shape}'
    assert keypoints.shape[0] > 1, f'There must be at least 2 keypoints, but received shape={keypoints.shape}'

    x = keypoints[:, 0]
    y = keypoints[:, 1]
    if keypoints.shape[1] == 3:
        c = keypoints[:, 2]
        x, y = x[c != 0], y[c != 0]

    x_left = np.min(x)
    y_up = np.min(y)

    x_right = np.max(x)
    y_down = np.max(y)
    return np.array([
        x_left, y_up,
        x_right, y_down
    ])
