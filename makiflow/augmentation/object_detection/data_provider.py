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

from __future__ import absolute_import
from makiflow.augmentation.base import Augmentor


class Data(Augmentor):
    def __init__(self, images, bboxes, classes):
        """
        Parameters
        ----------
        images : list
            List of images (ndarrays).
        bboxes : list
            List of ndarrays (concatenated bboxes [x1, y1, x2, y2]).
        classes : list
            List of ndarrays (arrays of the classes of the `bboxes`).
        """
        super().__init__()
        self.images = images
        self.bboxes = bboxes
        self.classes = classes
        self._img_shape = images[0].shape

    def get_data(self):
        return self.images, self.bboxes, self.classes

