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
from makiflow.core.debug_utils import d_msg
from .utils import decode_prediction
import numpy as np


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
        point_indicators_logits = []
        human_indicators_logits = []
        regressed_points = []
        for head in self._heads:
            point_indicators_logits += [head.get_point_indicators().get_data_tensor()]
            human_indicators_logits += [head.get_human_indicators().get_data_tensor()]
            regressed_points += [head.get_coords().get_data_tensor()]

        def flatten(x):
            b, h, w, c = x.get_shape().as_list()
            return tf.reshape(x, shape=[b, h * w, c])

        point_indicators_logits = list(map(flatten, point_indicators_logits))
        human_indicators_logits = list(map(flatten, human_indicators_logits))
        regressed_points = list(map(flatten, regressed_points))

        # If any of the lists is empty, it will be difficult to handle it using tf messages.
        # Hence this check is here.
        assert len(point_indicators_logits) != 0 and \
               len(human_indicators_logits) != 0 and \
               len(regressed_points) != 0, d_msg(
            self._name,
            'Length of the logits or regressed points is zero. '
            f'len(point_indicators_logits)={len(point_indicators_logits)}, '
            f'len(human_indicators_logits)={len(human_indicators_logits)}, '
            f'len(regressed_points)={len(regressed_points)}. '
            f'This is probably because the list of the heads is empty.'
        )

        # Concatenate the collected tensors
        self._point_indicators_logits = tf.concat(point_indicators_logits, axis=1)
        self._human_indicators_logits = tf.concat(human_indicators_logits, axis=1)
        regressed_points = tf.concat(regressed_points, axis=1)

        b, n, c = regressed_points.get_shape().as_list()
        w, h = self.get_image_size()
        regressed_points = tf.reshape(regressed_points, shape=[b, n, c // 2, 2])
        # Scale the grid: [-1, 1] -> [-w/2, w/2]
        regressed_points = regressed_points * np.array([w / 2, h / 2], dtype='float32')
        # Shift the grid: [-w/2, w/2] -> [0, w]
        regressed_points = regressed_points + np.array([w / 2, h / 2], dtype='float32')
        self._regressed_points = regressed_points
        # Used in predict
        self._point_indicators = tf.nn.sigmoid(self._point_indicators_logits)
        self._human_indicators = tf.nn.sigmoid(self._human_indicators_logits)

    def predict(self, X, min_conf=0.2, iou_th=0.5, raw_data=False):
        assert (self._session is not None)
        predictions = self._session.run(
            [self._regressed_points, self._point_indicators, self._human_indicators],
            feed_dict={self._in_x.get_data_tensor(): X}
        )
        if raw_data:
            return predictions

        processed_preds = []
        for coords, human_indicators, point_indicators in zip(*predictions):
            final_vectors = decode_prediction(
                prediction={'', (coords, human_indicators, point_indicators) },
                eps=min_conf,
                iou_th=iou_th
            )
            processed_preds.append(final_vectors)

        return processed_preds

    def get_heads(self):
        return self._heads


# CHECK MODEL'S PREDICT
if __name__ == '__main__':
    from .embedding_layer import SkeletonEmbeddingLayer

    # Generate points around a circle
    phi = np.linspace(0, 2 * np.pi, num=100)
    x = np.cos(phi) * 1.0 + [0]
    y = np.sin(phi) * 1.0 + [0]
    points = np.stack([x, y], axis=-1)

    from makiflow.layers import InputLayer

    # RUN A SANITY CHECK FIRST
    in_x = InputLayer(input_shape=[1, 3, 3, 100 * 2], name='offsets')
    # Never pass in a numpy array to the `custom_embedding` argument. Always use list.
    coords = SkeletonEmbeddingLayer(embedding_dim=None, name='TestEmbedding', custom_embedding=points)(in_x)

    print('Coords MakiTensor', coords)
    print('Coords TfTensor', coords.get_data_tensor())

    point_indicators = InputLayer(input_shape=[1, 3, 3, 100], name='point_indicators')
    human_indicators = InputLayer(input_shape=[1, 3, 3, 1], name='human_indicators')

    from .head import Head

    head = Head(coords, point_indicators, human_indicators)
    model = SSPModel(heads=[head], in_x=in_x)

    sess = tf.Session()
    model.set_session(sess)
    coords, _, _ = model._session.run(
        [model._regressed_points, model._point_indicators, model._human_indicators],
        feed_dict={
            model._in_x.get_data_tensor(): np.zeros(shape=[1, 3, 3, 200], dtype='float32'),
            point_indicators.get_data_tensor(): np.ones(shape=[1, 3, 3, 100], dtype='float32'),
            human_indicators.get_data_tensor(): np.ones(shape=[1, 3, 3, 1], dtype='float32')
        }
    )

    # Visualize the circles
    import matplotlib

    # For some reason matplotlib doesn't want to show the plot when it is called from PyCharm
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    coords = coords.reshape(-1, 2)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()
