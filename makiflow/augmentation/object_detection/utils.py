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
