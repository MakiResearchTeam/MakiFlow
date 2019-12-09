import numpy as np


def hor_flip_bboxes(bboxes, img_w):
    new_bboxs = np.copy(bboxes)
    new_bboxs[:, 0] = img_w - bboxes[:, 2]
    new_bboxs[:, 2] = img_w - bboxes[:, 0]
    return new_bboxs


def ver_flip_bboxes(bboxes, img_h):
    new_bboxs = np.copy(bboxes)
    new_bboxs[:, 1] = img_h - bboxes[:, 3]
    new_bboxs[:, 3] = img_h - bboxes[:, 1]
    return new_bboxs


def horver_flip_bboxes(bboxes, img_w, img_h):
    bboxes = hor_flip_bboxes(bboxes, img_w)
    bboxes = ver_flip_bboxes(bboxes, img_h)
    return bboxes
