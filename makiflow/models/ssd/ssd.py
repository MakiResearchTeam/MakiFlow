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
from makiflow.layers import InputLayer, ConcatLayer, ActivationLayer
from makiflow.core import MakiModel
from makiflow.models.ssd.ssd_utils import bboxes_wh2xy, bboxes_xy2wh
import numpy as np
import tensorflow as tf


class OffsetRegression:
    DUMMY = 0
    RCNN_LIKE = 1


class SSDModel(MakiModel):
    def get_feed_dict_config(self) -> dict:
        return {
            self._in_x: 0
        }

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, dcs: list, in_x: InputLayer, offset_reg_type=OffsetRegression.DUMMY, name='MakiSSD'):
        self._in_x = in_x

        self._dcs = dcs
        self._name = str(name)
        self.regression_type = offset_reg_type
        inputs = [in_x]
        graph_tensors = {}
        outputs = []
        for dc in dcs:
            # `confs` shape is [batch_sz, fmap_square, num_classes]
            # `offs` shape is [batch_sz, fmap_square, 4]
            confs, offs = dc.get_conf_offsets()
            graph_tensors.update(confs.get_previous_tensors())
            graph_tensors.update(offs.get_previous_tensors())
            graph_tensors.update(confs.get_self_pair())
            graph_tensors.update(offs.get_self_pair())

            outputs += [confs, offs]

        super().__init__(outputs, inputs, graph_tensors=graph_tensors)
        self._generate_default_boxes()
        self._prepare_inference_graph()

    # -------------------------------------------------------SETTING UP DEFAULT BOXES-----------------------------------

    def _generate_default_boxes(self):
        self.dboxes_wh = []
        # Also collect feature map sizes for later easy access to
        # particular bboxes
        self.dc_block_feature_map_sizes = []
        for dc in self._dcs:
            # [batch_sz, height, width, feature_maps]
            h, w = dc.get_height_width()
            dboxes = dc.get_dboxes()
            default_boxes = self._default_box_generator(self.input_shape[2], self.input_shape[1],
                                                        w, h, dboxes)
            self.dboxes_wh.append(default_boxes)
            self.dc_block_feature_map_sizes.append((h, w))

        self.dboxes_wh = np.vstack(self.dboxes_wh).astype(np.float32)
        # Converting default WH bboxes to XY format:
        # (x, y, w, h) -----> (x1, y1, x2, y2)
        self.dboxes_xy = bboxes_wh2xy(self.dboxes_wh)
        # Adjusting dboxes
        self._correct_dboxes()
        self.total_predictions = len(self.dboxes_xy)

    def _correct_dboxes(self):
        img_w = self.input_shape[2]
        img_h = self.input_shape[1]

        self.dboxes_xy[:, 0] = np.clip(self.dboxes_xy[:, 0], 0., img_w)  # x1
        self.dboxes_xy[:, 1] = np.clip(self.dboxes_xy[:, 1], 0., img_h)  # y1
        self.dboxes_xy[:, 2] = np.clip(self.dboxes_xy[:, 2], 0., img_w)  # x2
        self.dboxes_xy[:, 3] = np.clip(self.dboxes_xy[:, 3], 0., img_h)  # x3

        # Reassign WH dboxes to corrected values.
        self.dboxes_wh = bboxes_xy2wh(self.dboxes_xy)

    # noinspection PyMethodMayBeStatic
    def _default_box_generator(self, image_width, image_height, width, height, dboxes):
        """
        Generates default boxes in WH format.

        Parameters
        ----------
        image_width : int
            Width of the input image.
        image_height : int
            Height of the input height.
        width : int
            Width of the feature map.
        height : int
            Height of the feature map.
        dboxes : list
            List with default boxes characteristics (width, height). Example: [(1, 1), (0.5, 0.5)].

        Returns
        -------
        ndarray
            Array of 4d-vectors(np.arrays) contain characteristics of the default boxes in absolute
            coordinates: center_x, center_y, height, width.
        """
        box_count = width * height
        boxes_list = []

        width_per_cell = image_width / width
        height_per_cell = image_height / height

        for w, h in dboxes:
            boxes = np.zeros((box_count, 4))

            for i in range(height):
                current_height = i * height_per_cell
                for j in range(width):
                    current_width = j * width_per_cell
                    # (x, y) coordinates of the center of the default box
                    boxes[i * width + j][0] = current_width + width_per_cell / 2  # x
                    boxes[i * width + j][1] = current_height + height_per_cell / 2  # y
                    # (w, h) width and height of the default box
                    boxes[i * width + j][2] = width_per_cell * w  # width
                    boxes[i * width + j][3] = height_per_cell * h  # height
            boxes_list.append(boxes)


        return np.vstack(boxes_list)

    def get_dboxes_xy(self):
        return self.dboxes_xy

    def get_dboxes_wh(self):
        return self.dboxes_wh

    def get_dbox(self, dc_block_ind, dbox_category, x_pos, y_pos):
        dcblock_dboxes_to_pass = 0
        for i in range(dc_block_ind):
            dcblock_dboxes_to_pass += (
                    self.dc_block_feature_map_sizes[i][0] * self.dc_block_feature_map_sizes[i][1] *
                    len(self._dcs[i].get_dboxes())
            )
        for i in range(dbox_category):
            dcblock_dboxes_to_pass += (
                    self.dc_block_feature_map_sizes[dc_block_ind][0] * self.dc_block_feature_map_sizes[dc_block_ind][1]
            )
        dcblock_dboxes_to_pass += self.dc_block_feature_map_sizes[dc_block_ind][0] * x_pos
        dcblock_dboxes_to_pass += y_pos
        return self.dboxes_xy[dcblock_dboxes_to_pass]

    def _get_model_info(self):
        model_dict = {
            'name': self._name,
            'input_s': self._inputs[0].get_name(),
            'reg_type': self.regression_type,
            'dcs': []
        }

        for dc in self._dcs:
            model_dict['dcs'].append(dc.to_dict())
        return model_dict

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------SETTING UP INFERENCE OF THE MODEL--------------------------

    def _prepare_inference_graph(self):
        confidences = []
        offsets = []

        for dc in self._dcs:
            confs, offs = dc.get_conf_offsets()
            confidences += [confs]
            offsets += [offs]

        concatenate = ConcatLayer(axis=1, name='InferencePredictionConcat' + self._name)
        self.confidences_ish = concatenate(confidences)
        self.offsets = concatenate(offsets)
        self.offsets_tensor = self.offsets.get_data_tensor()
        # Dummy offset regression
        if self.regression_type == OffsetRegression.DUMMY:
            self.predicted_boxes = self.offsets_tensor + self.dboxes_xy
        # RCNN like offset regression
        elif self.regression_type == OffsetRegression.RCNN_LIKE:
            cx = self.dboxes_wh[:, 2] * self.offsets_tensor[:, :, 0] + self.dboxes_wh[:, 0]  # db_w * p_cx + db_cx
            cy = self.dboxes_wh[:, 3] * self.offsets_tensor[:, :, 1] + self.dboxes_wh[:, 1]  # db_h * p_cy + db_cy
            w_h = tf.exp(self.offsets_tensor[:, :, 2:])  # exponentiate width and height, magic math
            w = self.dboxes_wh[:, 2] * w_h[:, :, 0]  # times width
            h = self.dboxes_wh[:, 3] * w_h[:, :, 1]  # times height
            # [batch_sz, num_predictions] -> [batch_sz, num_predictions, 1]
            # Do this for the proper concatenation.
            cx = tf.expand_dims(cx, axis=-1)
            cy = tf.expand_dims(cy, axis=-1)
            w = tf.expand_dims(w, axis=-1)
            h = tf.expand_dims(h, axis=-1)
            predicted_bboxes_wh = tf.concat([cx, cy, w, h], axis=2)

            # Convert predicted bboxes to XY format
            up_x = predicted_bboxes_wh[:, :, 0] - predicted_bboxes_wh[:, :, 2] / 2.  # up_x
            up_y = predicted_bboxes_wh[:, :, 1] - predicted_bboxes_wh[:, :, 3] / 2.  # up_y
            bot_x = predicted_bboxes_wh[:, :, 0] + predicted_bboxes_wh[:, :, 2] / 2.  # bot_x
            bot_y = predicted_bboxes_wh[:, :, 1] + predicted_bboxes_wh[:, :, 3] / 2.  # bot_y
            up_x = tf.expand_dims(up_x, axis=-1)
            up_y = tf.expand_dims(up_y, axis=-1)
            bot_x = tf.expand_dims(bot_x, axis=-1)
            bot_y = tf.expand_dims(bot_y, axis=-1)
            self.predicted_boxes = tf.concat([up_x, up_y, bot_x, bot_y], axis=2)
        else:
            raise ValueError(f'Unknown offset regression type: {self.regression_type}')

        classificator = ActivationLayer(name='Classificator' + self._name, activation=tf.nn.softmax)
        self.confidences = classificator(self.confidences_ish)
        confidences_tensor = self.confidences.get_data_tensor()

        self.predictions = [confidences_tensor, self.predicted_boxes]

    def predict(self, X):
        assert (self._session is not None)
        return self._session.run(
            self.predictions,
            feed_dict={self._input_data_tensors[0]: X}
        )
