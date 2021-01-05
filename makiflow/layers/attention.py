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
from makiflow.core import MakiLayer, MakiRestorable, MakiBuilder
from .untrainable_layers import FlattenLayer
from .trainable_layers import ConvLayer
from .dev import ReshapeLikeLayer


def positional_encoding_v2(wh, dim, max_power=15):
    x_en = []
    y_en = []
    x_range = tf.range(start=0, limit=tf.cast(wh[0], 'float32'), dtype='float32') / tf.cast(wh[0], 'float32')
    y_range = tf.range(start=0, limit=tf.cast(wh[1], 'float32'), dtype='float32') / tf.cast(wh[1], 'float32')
    x, y = tf.meshgrid(x_range, y_range)
    for i in range(dim // 4):
        scale = tf.math.pow(2., max_power *(i / dim))
        x_en += [tf.sin(x * scale)]
        x_en += [tf.cos(x * scale)]
        y_en += [tf.sin(y * scale)]
        y_en += [tf.cos(y * scale)]
    return tf.stack(x_en + y_en, axis=-1)


class PositionalEncodingLayer(MakiLayer):
    DEPTH = 'depth'

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        depth = params[PositionalEncodingLayer.DEPTH]
        return PositionalEncodingLayer(name=name, depth=depth)

    def __init__(self, depth, name='PositionalEncodingLayer'):
        self._depth = depth
        super().__init__(name, [], [], {})

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                shape = tf.shape(x)
                h, w = shape[1], shape[2]
                pe = tf.expand_dims(positional_encoding_v2((w, h), self._depth), axis=0)
                x = x + pe
                return x

    def training_forward(self, X):
        return self.forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    def to_dict(self):
        return {
            MakiRestorable.TYPE: self.__class__.__name__,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                PositionalEncodingLayer.DEPTH: self._depth
            }
        }


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

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keys, queries, values = x
                # key and queries are assumed to have the same dimensionality
                dim = tf.cast(tf.shape(keys)[-1], 'float32')
                queries = tf.transpose(queries, perm=[0, 2, 1])
                keys = tf.nn.l2_normalize(keys, axis=-1)
                queries = tf.nn.l2_normalize(queries, axis=-1)

                attention_logits = tf.matmul(keys, queries) / tf.sqrt(dim)
                self.attention = tf.nn.softmax(attention_logits, axis=-1)

                output = tf.matmul(self.attention, values)
                return output

    def training_forward(self, X):
        return self.forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

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
    IN_F = 'in_f'
    KQ_DIM = 'kq_dim'

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        in_f = params[SpatialAttentionLayer.IN_F]
        kq_dim = params.get(SpatialAttentionLayer.KQ_DIM, 64)
        return SpatialAttentionLayer(name=name, in_f=in_f, kq_dim=kq_dim)

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
        self._positional_encoding = PositionalEncodingLayer(depth=kq_dim, name='enc' + name)
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

        self._to_grid = ReshapeLikeLayer('reshape' + name)
        self._flatten = FlattenLayer('flatten' + name, keep_depth=True)

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
        keys = self._flatten(keys)
        queries = self._flatten(queries)
        values = self._flatten(x)

        output = self._attention_head([keys, queries, values])
        output = self._to_grid([output, x])
        return super().__call__([x, output])

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return x[0] + x[1]

    def training_forward(self, X):
        return self.forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: self.__class__.__name__,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                self.IN_F: self._in_f,
                self.KQ_DIM: self._kq_dim
            }
        }


MakiBuilder.register_layers({
    PositionalEncodingLayer.__name__: PositionalEncodingLayer,
    AttentionLayer.__name__: AttentionLayer,
    SpatialAttentionLayer.__name__: SpatialAttentionLayer
})


if __name__ == '__main__':
    from makiflow.layers import InputLayer
    x = InputLayer(input_shape=[None, 32, 12, 64], name='name')
    x = SpatialAttentionLayer(in_f=64, name='attention')(x)
    print(x)
