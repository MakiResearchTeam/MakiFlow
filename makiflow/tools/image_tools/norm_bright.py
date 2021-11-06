import numpy as np

CONTRAST_SCALE = 1.0


def apply_op_norm_bright(img, contrast_scale=CONTRAST_SCALE, return_to_255_max_range=True):
    min_v, max_v = img.min(axis=(0,1)), img.max(axis=(0,1))
    cont_img = (img - min_v) / (max_v - min_v) * contrast_scale
    if return_to_255_max_range:
        return np.clip(cont_img * 255.0, 0.0, 255.0)
    return cont_img
