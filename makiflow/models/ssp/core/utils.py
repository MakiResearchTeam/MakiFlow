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


def aggregate_data(prediction):
    # Aggregate the data from the prediction
    levels = []
    human_indicators = []
    point_indicators = []
    for level, human_indicator, point_indicator in prediction:
        levels += [level]
        human_indicators += [human_indicator]
        point_indicators += [point_indicator]
    levels = np.concatenate(levels, axis=0)
    human_indicators = np.concatenate(human_indicators, axis=0)
    point_indicators = np.concatenate(point_indicators, axis=0)
    return levels, human_indicators, point_indicators


def decode_prediction(prediction, eps=0.1, iou_th=0.5, debug=False):
    coords, human_indicators, point_indicators = prediction
    coords, human_indicators, point_indicators = coords[0], human_indicators[0].reshape(-1), point_indicators[0]
    if debug:
        print('coords shape:', coords.shape)
        print('human_indicators shape:', human_indicators.shape)
        print('point_indicators shape:', point_indicators.shape)
    # Discard those vectors that does not contain any human
    to_pick = human_indicators > eps
    if debug:
        print('to_pick:', to_pick)
    coords = coords[to_pick]
    human_indicators = human_indicators[to_pick]
    point_indicators = point_indicators[to_pick]

    # Sort vectors, so that the most confident ones will be picked the first
    sorted_indices = np.argsort(human_indicators)
    coords = coords[sorted_indices]
    point_indicators = point_indicators[sorted_indices]

    coords = np.concatenate([coords, point_indicators], axis=-1)
    final_vectors = []
    while len(coords) != 0:
        # Pick the first `v` and add it to the set of final vectors
        v = coords[0]
        final_vectors.append(v)

        v_box = make_boxes([v])
        others_boxes = make_boxes(coords)

        ious = run(v_box, others_boxes).reshape(-1)
        # Discard similar ones
        to_discard = ious < iou_th
        if debug:
            print('\nious:', ious)
            print('others_boxes', others_boxes)
            print('coords before filtering: ', len(coords))
        coords = coords[to_discard]
        if debug:
            print('coords after filtering: ', len(coords))
    return final_vectors


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
