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

from abc import ABC
from copy import copy
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
# noinspection PyUnresolvedReferences
from tensorflow.nn import static_rnn, dynamic_rnn, bidirectional_dynamic_rnn, static_bidirectional_rnn

from makiflow.core.inference import MakiLayer, MakiTensor, MakiRestorable
from makiflow.layers.activation_converter import ActivationConverter


class NoCellStateException(Exception):
    ERROR_MSG = "The cells do not have a state yet. You might need to pass a MakiTensor into the " + \
        "__call__ method."

    def __init__(self):
        super().__init__(NoCellStateException.ERROR_MSG)


class CellType:
    BIDIR_DYNAMIC = 1
    BIDIR_STATIC = 2
    DYNAMIC = 3
    STATIC = 4

    @staticmethod
    def get_cell_type(bidirectional, dynamic):
        if bidirectional:
            if dynamic:
                return CellType.BIDIR_DYNAMIC
            else:
                return CellType.BIDIR_STATIC
        else:
            if dynamic:
                return CellType.DYNAMIC
            else:
                return CellType.STATIC


class RNNLayer(MakiLayer, ABC):
    TYPE = 'RNNLayer'

    def __init__(self, cells, params, named_params_dict, name, seq_length=None, dynamic=True,
                 bidirectional=False):
        self._cells = cells
        self._seq_length = seq_length
        self._dynamic = dynamic
        self._bidirectional = bidirectional
        self._cell_type = CellType.get_cell_type(bidirectional, dynamic)
        self._cells_state = None
        super().__init__(name, params, params, named_params_dict)

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        if self._cell_type == CellType.BIDIR_DYNAMIC:
            (outputs_f, outputs_b), (states_f, states_b) = \
                bidirectional_dynamic_rnn(cell_fw=self._cells, cell_bw=self._cells, inputs=x, dtype=tf.float32)
            # Creation of the two MakiTensors for both `outputs_f` and `outputs_b` is inappropriate since
            # the algorithm that builds the computational graph does not consider such case and
            # therefore can not handle this situation, it will cause an error.
            self._cells_state = tf.concat([states_f, states_b], axis=-1)
            return tf.concat([outputs_f, outputs_b], axis=-1)
        elif self._cell_type == CellType.BIDIR_STATIC:
            x = tf.unstack(x, num=self._seq_length, axis=1)
            outputs_fb, states_f, states_b = \
                static_bidirectional_rnn(cell_fw=self._cells, cell_bw=self._cells, inputs=x, dtype=tf.float32)
            self._cells_state = tf.concat([states_f, states_f], axis=-1)
            return outputs_fb
        elif self._cell_type == CellType.DYNAMIC:
            outputs, states = dynamic_rnn(self._cells, x, dtype=tf.float32)
            self._cells_state = states
            return outputs
        elif self._cell_type == CellType.STATIC:
            x = tf.unstack(x, num=self._seq_length, axis=1)
            outputs, states = static_rnn(self._cells, x, dtype=tf.float32)
            self._cells_state = states
            return tf.stack(outputs, axis=1)

    def training_forward(self, x):
        return self.forward(x)

    def get_cells(self):
        return self._cells

    def get_cells_state(self):
        if self._cells_state is None:
            raise NoCellStateException()
        return self._cells_state


class GRULayer(RNNLayer):
    TYPE = 'GRULayer'
    NUM_CELLS = 'num_cells'
    INPUT_DIM = 'input_dim'
    SEQ_LENGTH = 'seq_length'
    DYNAMIC = 'dynamic'
    BIDIRECTIONAL = 'bidirectional'
    ACTIVATION = 'activation'

    def __init__(self, num_cells, input_dim, seq_length, name, activation=tf.nn.tanh, dynamic=False,
                 bidirectional=False):
        """
        Parameters
        ----------
        num_cells : int
            Number of neurons in the layer.
        input_dim : int
            Dimensionality of the input vectors, e.t. number of features. Dimensionality
            example: [batch_size, seq_length, num_features(this is input_dim in this case)].
        seq_length : int
            Max length of the input sequences.
        activation : tensorflow function
            Activation function of the layer.
        dynamic : boolean
            Influences whether the layer will be working as dynamic RNN or static. The difference
            between static and dynamic is that in case of static TensorFlow builds static graph and the RNN
            will always go through each time step in the sequence. In case of dynamic TensorFlow will be
            creating RNN `in a while loop`, that is to say that using dynamic RNN you can pass sequences of
            variable length, but you have to provide list of sequences' lengthes. Currently API for using
            dynamic RNNs is not provided.
            WARNING! THIS PARAMETER DOESN'T PLAY ANY ROLE IF YOU'RE GONNA STACK RNN LAYERS.
        bidirectional : boolean
            Influences whether the layer will be bidirectional.
            WARNING! THIS PARAMETER DOES NOT PLAY ANY ROLE IF YOU ARE GOING TO STACK RNN LAYERS.
        """
        self._num_cells = num_cells
        self._input_dim = input_dim
        self._f = activation
        cell = GRUCell(num_units=num_cells, activation=activation, dtype=tf.float32)
        cell.build(input_shape=[None, tf.Dimension(self._input_dim)])
        params = cell.variables
        param_common_name = name + f'_{num_cells}_{input_dim}_{seq_length}'
        named_params_dict = {(param_common_name + '_' + str(i)): param for i, param in enumerate(params)}
        super().__init__(
            cells=cell,
            params=params,
            named_params_dict=named_params_dict,
            name=name,
            seq_length=seq_length,
            dynamic=dynamic,
            bidirectional=bidirectional
        )

    @staticmethod
    def build(params: dict):
        num_cells = params[GRULayer.NUM_CELLS]
        input_dim = params[GRULayer.INPUT_DIM]
        seq_length = params[GRULayer.SEQ_LENGTH]
        name = params[MakiRestorable.NAME]
        dynamic = params[GRULayer.DYNAMIC]
        bidirectional = params[GRULayer.BIDIRECTIONAL]
        activation = ActivationConverter.str_to_activation(params[GRULayer.ACTIVATION])
        return GRULayer(
            num_cells=num_cells,
            input_dim=input_dim,
            seq_length=seq_length,
            name=name,
            activation=activation,
            dynamic=dynamic,
            bidirectional=bidirectional
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: GRULayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                GRULayer.NUM_CELLS: self._num_cells,
                GRULayer.INPUT_DIM: self._input_dim,
                GRULayer.SEQ_LENGTH: self._seq_length,
                GRULayer.DYNAMIC: self._dynamic,
                GRULayer.BIDIRECTIONAL: self._bidirectional,
                GRULayer.ACTIVATION: ActivationConverter.activation_to_str(self._f)
            }
        }


class LSTMLayer(MakiLayer):
    TYPE = 'LSTMLayer'
    NUM_CELLS = 'num_cells'
    INPUT_DIM = 'input_dim'
    SEQ_LENGTH = 'seq_length'
    DYNAMIC = 'dynamic'
    BIDIRECTIONAL = 'bidirectional'
    ACTIVATION = 'activation'

    OUTPUT_HIDDEN_STATE = 'HIDDEN_STATE'
    OUTPUT_LAST_CANDIDATE = 'LAST_CANDIDATE'
    OUTPUT_LAST_HIDDEN_STATE = 'LAST_HIDDEN_STATE'

    def __init__(self, in_d, out_d, name, activation=tf.nn.tanh, dynamic=True):
        """
        Parameters
        ----------
        in_d : int
            Number of neurons in the layer.
        out_d : int
            Dimensionality of the input vectors, e.t. number of features. Dimensionality:
            [batch_size, seq_length, num_features(this is input_dim in this case)].
        seq_length : int
            Max length of the input sequences.
        activation : tensorflow function
            Activation function of the layer.
        dynamic : boolean
            Influences whether the layer will be working as dynamic RNN or static. The difference
            between static and dynamic is that in case of static TensorFlow builds static graph and the RNN
            will always go through each time step in the sequence. In case of dynamic TensorFlow will be
            creating RNN `in a while loop`, that is to say that using dynamic RNN you can pass sequences of
            variable length, but you have to provide list of sequences' lengthes. Currently API for using
            dynamic RNNs is not provided.
            WARNING! THIS PARAMETER DOESN'T PLAY ANY ROLE IF YOU'RE GONNA STACK RNN LAYERS.
        """
        self._num_cells = in_d
        self._input_dim = in_d
        self._f = activation
        self._cell = LSTMCell(num_units=out_d, activation=activation, dtype=tf.float32)
        self._cell.build(input_shape=[out_d])
        self._dynamic = dynamic
        params = self._cell.variables
        param_common_name = name + f'_{in_d}_{out_d}'
        named_params_dict = {(param_common_name + '_' + str(i)): param for i, param in enumerate(params)}
        super().__init__(
            name=name,
            params=params,
            regularize_params=params,
            named_params_dict=named_params_dict,
            outputs_names=[
                LSTMLayer.OUTPUT_HIDDEN_STATE,
                LSTMLayer.OUTPUT_LAST_CANDIDATE,
                LSTMLayer.OUTPUT_LAST_HIDDEN_STATE
            ]
        )

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        if self._dynamic:
            dynamic_x = dynamic_rnn(self._cell, x, dtype=tf.float32)
            # hidden states, (last candidate value, last hidden state)
            hs, (c_last, h_last) = dynamic_x
            return hs, c_last, h_last
        else:
            unstack_x = tf.unstack(x, axis=1)
            static_x = static_rnn(self._cell, unstack_x, dtype=tf.float32)
            hs_list, (c_last, h_last) = static_x
            hs = tf.stack(hs_list, axis=1)
            return hs, c_last, h_last

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        num_cells = params[LSTMLayer.NUM_CELLS]
        input_dim = params[LSTMLayer.INPUT_DIM]
        seq_length = params[LSTMLayer.SEQ_LENGTH]
        name = params[MakiRestorable.NAME]
        dynamic = params[LSTMLayer.DYNAMIC]
        bidirectional = params[LSTMLayer.BIDIRECTIONAL]
        activation = ActivationConverter.str_to_activation(params[LSTMLayer.ACTIVATION])
        return LSTMLayer(
            in_d=num_cells,
            out_d=input_dim,
            seq_length=seq_length,
            name=name,
            activation=activation,
            dynamic=dynamic,
            bidirectional=bidirectional
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: LSTMLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                LSTMLayer.NUM_CELLS: self._num_cells,
                LSTMLayer.INPUT_DIM: self._input_dim,
                LSTMLayer.SEQ_LENGTH: self._seq_length,
                LSTMLayer.DYNAMIC: self._dynamic,
                LSTMLayer.BIDIRECTIONAL: self._bidirectional,
                LSTMLayer.ACTIVATION: ActivationConverter.activation_to_str(self._f)
            }
        }


class RNNBlock(RNNLayer):
    TYPE = 'RNNBlock'
    SEQ_LENGTH = 'seq_length'
    DYNAMIC = 'dynamic'
    BIDIRECTIONAL = 'bidirectional'
    RNN_LAYERS = 'rnn_layers'

    def __init__(self, rnn_layers, seq_length, name, dynamic=False, bidirectional=False):
        """
        Parameters
        ----------
        rnn_layers : list
            List of RNN layers to stack.
        seq_length : int
            Max length of the input sequences.
        dynamic : boolean
            Influences whether the layer will be working as dynamic RNN or static. The difference
            between static and dynamic is that in case of static TensorFlow builds static graph and the RNN
            will always go through each time step in the sequence. In case of dynamic TensorFlow will be
            creating RNN `in a while loop`, that is to say that using dynamic RNN you can pass sequences of
            variable length, but you have to provide list of sequences' lengthes. Currently API for using
            dynamic RNNs is not provided.
        bidirectional : boolean
            Influences whether the layer will be bidirectional.
        """
        self._rnn_layers = rnn_layers
        rnn_cells = []
        for layer in rnn_layers:
            rnn_cells.append(layer.get_cells())
        stacked_cells = MultiRNNCell(cells=rnn_cells)

        params = []
        named_params_dict = {}
        for layer in rnn_layers:
            params += layer.get_params()
            named_params_dict.update(layer.get_params_dict())

        super().__init__(
            cells=stacked_cells,
            params=params,
            named_params_dict=named_params_dict,
            name=name,
            seq_length=seq_length,
            dynamic=dynamic,
            bidirectional=bidirectional
        )

    @staticmethod
    def build(params: dict):
        seq_length = params[RNNBlock.SEQ_LENGTH]
        dynamic = params[RNNBlock.DYNAMIC]
        bidirectional = params[RNNBlock.BIDIRECTIONAL]

        rnn_layers_info = params[RNNBlock.RNN_LAYERS]
        rnn_layers = []
        for i in range(len(rnn_layers_info)):
            single_layer = rnn_layers_info[i]
            single_params = single_layer[MakiRestorable.PARAMS]
            single_type = single_layer[RNNBlock.FIELD_TYPE]
            rnn_layers.append(RNNLayerAddress.ADDRESS_TO_CLASSES[single_type].build(single_params))

        return RNNBlock(
            rnn_layers=rnn_layers,
            seq_length=seq_length,
            dynamic=dynamic,
            bidirectional=bidirectional
        )

    def to_dict(self):
        rnnblock_dict = {
            MakiRestorable.FIELD_TYPE: RNNBlock.TYPE ,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                RNNBlock.SEQ_LENGTH: self._seq_length,
                RNNBlock.DYNAMIC: self._dynamic,
                RNNBlock.BIDIRECTIONAL: self._bidirectional,
            }
        }

        rnn_layers_dict = {
            RNNBlock.RNN_LAYERS: []
        }
        for layer in self._rnn_layers:
            rnn_layers_dict[RNNBlock.RNN_LAYERS].append(layer.to_dict())

        rnnblock_dict.update(rnn_layers_dict)
        return rnnblock_dict


class EmbeddingLayer(MakiLayer):
    TYPE = 'EmbeddingLayer'
    NUM_EMBEDDINGS = 'num_embeddings'
    DIM = 'dim'

    def __init__(self, num_embeddings, dim, name):
        """
        Parameters
        ----------
        num_embeddings : int
            Number of embeddings in the embedding matrix(e.g. size of the vocabulary in case of word embedding).
        dim : int
            Dimensionality of the embedding.
        name : string or anything convertable to string
            Name of the layer.
        """
        self._num_embeddings = num_embeddings
        self._dim = dim
        name = 'Embedding_' + str(name)
        embed = np.random.randn(num_embeddings, dim) * np.sqrt(12 / (num_embeddings + dim))
        self.embed = tf.Variable(embed, name=name, dtype=tf.float32)

        params = [self.embed]
        named_params_dict = {name: self.embed}
        super().__init__(name, params, named_params_dict)

    def forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        return tf.nn.embedding_lookup(self.embed, x)

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        num_embeddings = params[EmbeddingLayer.NUM_EMBEDDINGS]
        dim = params[EmbeddingLayer.DIM]
        name = params[MakiRestorable.NAME]
        return EmbeddingLayer(
            num_embeddings=num_embeddings,
            dim=dim,
            name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE:  EmbeddingLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                EmbeddingLayer.NUM_EMBEDDINGS: self._num_embeddings,
                EmbeddingLayer.DIM: self._dim,
            }
        }


class RNNLayerAddress:

    ADDRESS_TO_CLASSES = {
        RNNLayer.TYPE: RNNLayer,
        GRULayer.TYPE: GRULayer,
        LSTMLayer.TYPE: LSTMLayer,
        RNNBlock.TYPE: RNNBlock,
        EmbeddingLayer.TYPE: EmbeddingLayer,
    }


from makiflow.core.inference.maki_builder import MakiBuilder

MakiBuilder.register_layers(RNNLayerAddress.ADDRESS_TO_CLASSES)

del MakiBuilder

