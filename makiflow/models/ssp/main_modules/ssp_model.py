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
from makiflow.layers import InputLayer
from makiflow.base.maki_entities import MakiCore
from .class_reg_head import ClassRegHead
import tensorflow as tf
from .ssp_interface import SSPInterface


class SSPModel(MakiCore, SSPInterface):
    def get_heads(self):
        return self._heads

    def _get_model_info(self):
        # TODO
        pass

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, cr_heads: list, in_x: InputLayer, name='MakiSSD'):
        self._crh = cr_heads
        self._name = str(name)
        inputs = [in_x]

        graph_tensors = {}
        outputs = []
        self._heads = {}
        for head in cr_heads:
            c_logits = head.get_classification_logits()
            hp_logits = head.get_human_presence_logits()
            p_offsets = head.get_points_offsets()

            outputs += [c_logits]
            outputs += [hp_logits]
            outputs += [p_offsets]

            graph_tensors.update(c_logits.get_previous_tensors())
            graph_tensors.update(hp_logits.get_previous_tensors())
            graph_tensors.update(p_offsets.get_previous_tensors())
            graph_tensors.update(c_logits.get_self_pair())
            graph_tensors.update(hp_logits.get_self_pair())
            graph_tensors.update(p_offsets.get_self_pair())

        super().__init__(graph_tensors, outputs, inputs)

        # Create tensors that will be ran in predict method
        self._setup_inference()

    def _setup_inference(self):
        _, H, W, _ = self._inputs[0].get_shape()
        # Collect tensors from every head.
        classification_logits = []
        human_presence_logits = []
        regressed_points = []
        for head_name in self._heads:
            head = self._heads[head_name]
            head_tensors = head.get_tensor_dict()
            classification_logits += [head_tensors[ClassRegHead.CLASS_LOGITS].get_data_tensor()]
            human_presence_logits += [head_tensors[ClassRegHead.HUMANI_LOGITS].get_data_tensor()]
            offsets = head_tensors[ClassRegHead.POINTS_OFFSETS]
            regressed_points += [head.get_regressed_points_tensor(offsets, (H, W))]

        # Concatenate the collected tensors
        self._classification_logits = tf.concat(classification_logits, axis=1)
        self._human_presence_logits = tf.concat(human_presence_logits, axis=1)

        # Used in predict
        self._regressed_points = tf.concat(regressed_points, axis=1)
        self._classification_vals = tf.nn.softmax(self._classification_logits, axis=-1)
        self._human_presence_indicators = tf.nn.sigmoid(self._human_presence_logits)

    def predict(self, X):
        assert (self._session is not None)
        return self._session.run(
            [self._regressed_points, self._classification_vals, self._human_presence_indicators],
            feed_dict={self._input_data_tensors[0]: X}
        )


