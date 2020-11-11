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
import tensorflow as tf
from .ssp_interface import SSPInterface


class SSPModel(SSPInterface):
    def get_feed_dict_config(self) -> dict:
        return {
            self._in_x: 0
        }

    def _get_model_info(self):
        # TODO
        pass

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, in_x: InputLayer, heads: list, name='MakiSSP'):
        self._in_x = in_x
        self._heads = heads
        self._name = str(name)
        inputs = [in_x]

        outputs = []
        self._heads = {}
        for head in heads:
            coords = head.get_coords()
            point_indicators = head.get_point_indicators()
            human_indicators = head.get_human_indicators()

            outputs += [coords]
            outputs += [point_indicators]
            outputs += [human_indicators]

        super().__init__(outputs, inputs)

        # Create tensors that will be ran in predict method
        self._setup_inference()

    def get_image_size(self):
        _, h, w, _ = self._in_x.get_shape()
        return w, h

    def _setup_inference(self):
        # Collect tensors from every head.
        classification_logits = []
        human_presence_logits = []
        regressed_points = []
        for head in self._heads:
            classification_logits += [head.get_point_indicators().get_data_tensor()]
            human_presence_logits += [head.get_human_indicators().get_data_tensor()]
            regressed_points += [head.get_coords().get_data_tensor()]

        def flatten(x):
            b, h, w, c = x.get_shape().as_list()
            return tf.reshape(x, shape=[b, h * w, c])

        classification_logits   = list(map(flatten, classification_logits))
        human_presence_logits   = list(map(flatten, human_presence_logits))
        regressed_points        = list(map(flatten, regressed_points))

        # Concatenate the collected tensors
        self._classification_logits = tf.concat(classification_logits, axis=1)
        self._human_presence_logits = tf.concat(human_presence_logits, axis=1)
        regressed_points            = tf.concat(regressed_points, axis=1)

        b, n, c = regressed_points.get_shape().as_list()
        self._regressed_points = tf.reshape(regressed_points, shape=[b, n, c // 2, 2])

        # Used in predict
        self._classification_vals = tf.nn.softmax(self._classification_logits, axis=-1)
        self._human_presence_indicators = tf.nn.sigmoid(self._human_presence_logits)

    def predict(self, X):
        assert (self._session is not None)
        return self._session.run(
            [self._regressed_points, self._classification_vals, self._human_presence_indicators],
            feed_dict={self._input_data_tensors[0]: X}
        )

    def get_heads(self):
        return self._heads

