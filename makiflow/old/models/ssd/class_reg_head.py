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

from makiflow.core import MakiTensor
from makiflow.layers import ConvLayer
from makiflow.layers import ReshapeLayer
import tensorflow as tf
import numpy as np


class ClassRegHead:
    """
    This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """
    DEFAULT_BOXES = [
        (1, 1), (1, 2), (2, 1),
        (1.26, 1.26), (1.26, 2.52), (2.52, 1.26),
        (1.59, 1.59), (1.59, 3.17), (3.17, 1.59)
    ]
    CLASS_LOGITS = 'CLASS_LOGITS'
    BOXES_OFFSETS = 'POINTS_OFFSETS'

    def __init__(
            self,
            reg_f: MakiTensor,
            class_f: MakiTensor,
            name,
            n_classes: int,
            default_boxes: list = None,
            reg_init_type='he', class_init_type='he',
            reg_kernel=(1, 1), class_kernel=(1, 1)
    ):
        """
        Parameters
        ----------
        reg_f : MakiTensor
            Source of features for the bbox regressor.
        class_f : MakiTensor
            Source of features for the classificator.
        default_boxes : ndarray of shape (n_points, 2)
            Default points of the skeleton. Their coordinates (x, y) must be centered and normalized
            in the [-1, 1] interval.
        name : str or int
            Will be used as conjugation for the names of the classificator and regressor.
        """
        self._reg_f = reg_f
        self._class_f = class_f
        self._default_boxes = ClassRegHead.DEFAULT_BOXES
        if default_boxes is not None:
            self._default_boxes = default_boxes
        self._n_classes = n_classes

        self.name = str(name)
        self.reg_init_type = reg_init_type
        self.class_init_type = class_init_type
        self.reg_kernel = reg_kernel
        self.class_kernel = class_kernel

        self._check_dimensionality()
        self._setup_heads()
        self._make_detections()

    def _check_dimensionality(self):
        # Height and width of the feature sources must be the same
        _, CH, CW, _ = self._class_f.shape()
        _, RH, RW, _ = self._reg_f.shape()
        msg = 'Dimensionality of {0} and {1} are not the same. Dim of {0} is {2}, dim of {1} is {3}'
        assert CH == RH and CW == RW, msg.format('class_f', 'reg_f', (CH, CW), (RH, RW))

    def _setup_heads(self):
        # SETUP CLASSIFICATION HEAD
        # Class for each point + class which indicates presence of a human in the bounding box
        B, H, W, C = self._class_f.shape()
        kw, kh = self.class_kernel
        self._classification_head = ConvLayer(
            kw=kw, kh=kh, in_f=C, out_f=self._n_classes,
            activation=None, padding='SAME', kernel_initializer=self.class_init_type,
            name='PointsClassifier/' + str(self.name)
        )

        # SETUP REGRESSION HEAD
        n_box_types = len(self._default_boxes)
        B, H, W, C = self._reg_f.shape()
        kw, kh = self.reg_kernel
        self._regression_head = ConvLayer(
            kw=kw, kh=kh, in_f=C, out_f=n_box_types * 4,     # regression of x, y, w, h simultaneously
            activation=None, padding='SAME', kernel_initializer=self.reg_init_type,
            name='PointsRegressor/' + str(self.name)
        )

    def _make_detections(self):
        """
        Creates list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences, offsets]
        """
        self._classification_logits = self._classification_head(self._class_f)
        _, H, W, C = self._classification_logits.shape()
        self._flat_classification_logits = ReshapeLayer(new_shape=[H * W, C], name=f'{self.name}/flat_class')(
            self._classification_logits
        )

        self._points_offsets = self._regression_head(self._reg_f)

    def get_classification_logits(self):
        return self._flat_classification_logits

    def get_points_offsets(self):
        return self._points_offsets

    def get_regressed_points_tensor(self, bbox_offsets, image_shape):
        """
        Applies offsets to the skeleton points and returns ready to use coordinates.
        This function is needed to create separate training and inference tensors for points regression later.

        Parameters
        ----------
        bbox_offsets : MakiTensor
            Tensor of the offsets that will be applied to shift the default points.
        image_shape : tuple of two ints
            Contains width and height of the image. (width, height)

        Returns
        -------
        tf.Tensor of shape [batch_sz, n_features, n_points * 2]
            Points coordinates with applied offsets.
        """
        box_types = self._default_boxes
        B, H, W, C = self._points_offsets.shape()
        B_, H_, W_, C_ = bbox_offsets.shape()
        assert (B, H, W, C) == (B_, H_, W_, C_), f'{self.name} / Original and new shapes must be the same.' \
                                                 f' Original={(B, H, W, C)}, new={(B_, H_, W_, C_)}'

        im_w, im_h = image_shape
        cell_h = H / im_h
        cell_w = W / im_w
        x = np.arange(cell_w / 2, im_w, cell_w)
        y = np.arange(cell_h / 2, im_h, cell_h)
        x_map, y_map = np.meshgrid(x, y)
        xy_map = np.stack([x_map, y_map], axis=-1)

        w_map = np.ones(shape=[H, W, 1]) * cell_w / 2
        h_map = np.ones(shape=[H, W, 1]) * cell_h / 2

        xy_wh_map = np.concatenate([xy_map, w_map, h_map], axis=-1)
        xy_wh_map = np.expand_dims(xy_wh_map, axis=0)

        xy_wh_maps = []
        for box_type in box_types:
            xy_wh_map_ = xy_wh_map.copy()
            xy_wh_map_[:, :, 0:2] *= np.array(box_type)
            xy_wh_maps += [xy_wh_map_]

        xy_wh_maps = np.concatenate(xy_wh_maps, axis=-1)
        bbox_offsets_tensor = bbox_offsets.tensor()
        return bbox_offsets_tensor + xy_wh_maps

    def get_name(self):
        return self.name

    def get_tensor_dict(self):
        return {
            ClassRegHead.CLASS_LOGITS: self.get_classification_logits(),
            ClassRegHead.BOXES_OFFSETS: self.get_points_offsets()
        }
