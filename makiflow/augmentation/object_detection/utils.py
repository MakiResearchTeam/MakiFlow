import numpy as np


def hor_flip_bboxes(bboxes, img_w):
    new_bboxs = np.copy(bboxes)
    new_bboxs[:, 0] = img_w - bboxes[:, 2]
    new_bboxs[:, 2] = img_w - bboxes[:, 0]
    return new_bboxs


def ver_flip_bboxes(bboxes, img_h):
    pass


def horver_flip_bboxes(bboxes, img_w, img_h):
    pass
