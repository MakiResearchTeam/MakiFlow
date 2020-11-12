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
    assert keypoints.shape[1] < 4, f'keypoints are not 2-dimensional. Received shape={keypoints.shape}'
    assert keypoints.shape[0] > 1, f'There must be at least 2 keypoints, but received shape={keypoints.shape}'

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    if keypoints.shape[1] == 3:
        c = keypoints[:, 2]
        x, y = x[c != 0], y[c != 0]

    if len(x) == 0:
        # If the human is absent, all of its points marked as absent
        return np.array([
            0., 0.,
            0., 0.
        ])

    x_left = np.min(x)
    y_up = np.min(y)

    x_right = np.max(x)
    y_down = np.max(y)
    return np.array([
        x_left, y_up,
        x_right, y_down
    ])


def make_boxes(keypoints):
    boxes = []
    for k in keypoints:
        #print(k.shape)
        boxes += [make_box(k)]
    return np.stack(boxes, axis=0)


def run(bboxes1, bboxes2):
    """
    Calculates IoU between each box in `bboxes1` and each box in `bboxes2`.
    Parameters
    ----------
    bboxes1 : ndarray of shape [n, 4]
    bboxes2 : ndarray of shape [m, 4]

    Returns
    -------
    ndarray of shape [n, m]
        (i,j) position contains IoU value between ith box from `bboxes1` and jth box from `bboxes2`.
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def generate_grid(w, h, dim):
    delta_x = 1 / w
    x = np.linspace(start=-1 + delta_x, stop=1 - delta_x, num=w, dtype='float32')

    delta_y = 1 / h
    y = np.linspace(start=-1 + delta_y, stop=1 - delta_y, num=h, dtype='float32')
    # Has shape [h, w, 2]
    grid = np.stack(np.meshgrid(x, y), axis=-1)
    stacked_grid = np.stack([grid] * dim)
    stacked_grid = stacked_grid.transpose([1, 2, 0, 3])
    return stacked_grid


def generate_level_stacked(size, embedding):
    """
    size = [w, h]
    embedding - [n, 2]
    """
    w, h = size
    # level - [h, w, 2]
    level = generate_grid(w, h)
    # stacked_level - [depth, h, w, 2]
    depth = embedding.shape[0]
    stacked_level = np.stack([level] * depth)
    # stacked_level - [h, w, depth, 2]
    stacked_level = stacked_level.transpose([1, 2, 0, 3])

    # embedding - [1, 1, n, 2]
    embedding = embedding.reshape(1, 1, -1, 2)
    # Normalize the embedding
    embedding = embedding / np.array([w, h])
    return stacked_level + embedding
