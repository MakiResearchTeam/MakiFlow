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

import tensorflow as tf
from makiflow.base import MakiLayer, MakiRestorable
from .untrainable_layers import ReshapeLayer
from .trainable_layers import ConvLayer


def positional_encoding_v2(wh, dim, max_power=15):
    x_en = []
    y_en = []
    x_range = tf.range(start=0, limit=wh[0], dtype='float32') / float(wh[0])
    y_range = tf.range(start=0, limit=wh[1], dtype='float32') / float(wh[1])
    x, y = tf.meshgrid(x_range, y_range)
    den = max(wh)
    for i in range(dim // 4):
        scale = tf.math.pow(2., max_power *(i / dim))
        x_en += [tf.sin(x * scale)]
        x_en += [tf.cos(x * scale)]
        y_en += [tf.sin(y * scale)]
        y_en += [tf.cos(y * scale)]
    return tf.stack(x_en + y_en, axis=-1)


class PositionalEncodingLayer(MakiLayer):
    @staticmethod
    def build(params: dict):
        # TODO
        pass

    def __init__(self, name='PositionalEncodingLayer'):
        super().__init__(name, [], [], {})

    def _forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                _, h, w, d = x.get_shape().as_list()
                pe = tf.expand_dims(positional_encoding_v2((h, w), d), axis=0)
                x = x + pe
                return x

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    def to_dict(self):
        return {}


class AttentionLayer(MakiLayer):
    TYPE = 'AttentionLayer'
    L2_NORMALIZE = 'L2_NORMALIZE'
    DIM_NORMALIZE = 'DIM_NORMALIZE'

    @staticmethod
    def build(params: dict):
        return AttentionLayer(
            name=params[MakiRestorable.NAME],
            l2_normalize=params[AttentionLayer.L2_NORMALIZE],
            dim_normalize=params[AttentionLayer.DIM_NORMALIZE]
        )

    def __init__(self, name, l2_normalize=True, dim_normalize=True):
        """
        Uses an attention mechanism similar to the one used in Transformers.
        Parameters
        ----------
        name : str
            Name of the layer.
        l2_normalize : bool
            If set to True every vector in keys and queries will be normalized.
        dim_normalize : bool
            If set to True the result of the dot product between keys and queries will
            be divided by the dimensionality of the keys and the queries.
        """
        self._l2_normalize = l2_normalize
        self._dim_normalize = dim_normalize
        super().__init__(name, [], [], {})

    def __call__(self, x):
        """
        Parameters
        ----------
        x : list
            List containing MakiTensors for keys, queries and values.

        Returns
        -------
        MakiTensor
        """
        return super().__call__(x)

    def _forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keys, queries, values = x
                # key and queries are assumed to have the same dimensionality
                dim = float(int(keys.get_shape()[-1]))
                queries = tf.transpose(queries, perm=[0, 2, 1])
                keys = tf.nn.l2_normalize(keys, axis=-1)
                queries = tf.nn.l2_normalize(queries, axis=-1)

                attention_logits = tf.matmul(keys, queries) / tf.sqrt(dim)
                self.attention = tf.nn.softmax(attention_logits, axis=-1)

                output = tf.matmul(self.attention, values)
                return output

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    def to_dict(self):
        return {
            MakiRestorable.TYPE: AttentionLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                AttentionLayer.L2_NORMALIZE: self._l2_normalize,
                AttentionLayer.DIM_NORMALIZE: self._dim_normalize
            }
        }


class SpatialAttentionLayer(MakiLayer):
    @staticmethod
    def build(params: dict):
        # TODO
        pass

    def __init__(self, name, in_f, kq_dim=64):
        """
        Spatial attention mechanism made for CNNs.
        Parameters
        ----------
        name : str
            Name of the layer.
        in_f : int
            Dimensionality of the vectors in the input feature map.
        kq_dim : int
            Dimensionality of the vectors of the keys and the queries.
        """
        self._kq_dim = kq_dim
        self._in_f = in_f
        self._positional_encoding = PositionalEncodingLayer(name='enc' + name)
        self._queries_projection = ConvLayer(
            kw=1,
            kh=1,
            in_f=in_f,
            out_f=kq_dim,
            name='queries_projection' + name,
            activation=None,
        )

        self._keys_projection = ConvLayer(
            kw=1,
            kh=1,
            in_f=in_f,
            out_f=kq_dim,
            name='keys_projection' + name,
            activation=None
        )

        self._attention_head = AttentionLayer(name='attention' + name)
        super().__init__(name, [], [], {})

    def get_qk_projections(self):
        """
        Returns
        -------
        tuple of MakiLayers
            Projection layers for the keys and the queries.
        """
        return self._queries_projection, self._keys_projection

    def get_attention_head(self):
        """
        Returns
        -------
        MakiLayer
            The attention layer.
        """
        return self._attention_head

    def __call__(self, x):
        # Project keys and queries to lower dimensional space.
        keys = self._keys_projection(x)
        queries = self._queries_projection(x)
        # Use the positional encoding.
        keys = self._positional_encoding(keys)
        queries = self._positional_encoding(queries)

        # Flatten the feature maps.
        b, h, w, c = x.get_shape()
        self._flatten_kq = ReshapeLayer(
            new_shape=[b, h * w, self._kq_dim], name=self.get_name() + '/FlattenKQ'
        )
        keys = self._flatten_kq(keys)
        queries = self._flatten_kq(queries)

        self._flatten_x = ReshapeLayer(
            new_shape=[b, h * w, c], name=self.get_name() + '/FlattenV'
        )
        values = self._flatten_x(x)

        output = self._attention_head([keys, queries, values])
        self._to_grid = ReshapeLayer(
            new_shape=[b, h, w, c], name=self.get_name() + '/ToGrid'
        )
        output = self._to_grid(output)
        return super().__call__([x, output])

    def _forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return x[0] + x[1]

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    def to_dict(self):
        return {}
