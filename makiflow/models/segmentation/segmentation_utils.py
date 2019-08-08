import numpy as np
from tensorflow import logging
from skimage.segmentation import find_boundaries


# Implementation taken from https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
def create_weight_map(masks: np.ndarray, w0=10, sigma=5) -> np.ndarray:
    masks_count, height, width = masks.shape[:3]
    masks = masks.astype(int)

    dist_map = np.zeros((height * width, masks_count))

    x1, y1 = np.meshgrid(np.arange(width), np.arange(height))
    x1, y1 = np.column_stack((x1.reshape(-1), y1.reshape(-1))).T

    for i, mask in enumerate(masks):
        bounds = find_boundaries(mask, mode='inner')
        x2, y2 = np.nonzero(bounds)
        x_sum = (x2.reshape(-1, 1) - x1.reshape(1, -1)) ** 2
        y_sum = (y2.reshape(-1, 1) - y1.reshape(1, -1)) ** 2
        dist_map[:, i] = np.sqrt(x_sum, y_sum).min(axis=0)

    ix = np.arange(dist_map.shape[0])
    if dist_map.shape[1] == 1:
        d1 = dist_map.reshape(-1)
        border_loss_map = w0 * np.exp((-1 * d1 ** 2) / (2 * (sigma ** 2)))
    else:
        if dist_map.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(dist_map, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(dist_map, 2, axis=1)[:, :2].T
        d1 = dist_map[ix, d1_ix]
        d2 = dist_map[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

    x_border_loss = np.zeros((height, width))
    x_border_loss[x1, y1] = border_loss_map
    loss = np.zeros((height, width))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    zz = x_border_loss + loss
    return zz


def create_weight_map_based_on_frequency(masks: np.ndarray) -> np.ndarray:
    for mask in masks:
        uniq, counts = np.unique(mask, return_counts=True)
        for index, num in enumerate(uniq):
            if num != 0:
                mapped = (mask == num)
                mask[mapped] = 1 - (counts[index] / mask.size)
    return masks
