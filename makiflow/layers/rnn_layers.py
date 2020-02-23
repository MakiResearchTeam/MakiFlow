from __future__ import absolute_import

from abc import ABC

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
# noinspection PyUnresolvedReferences
from tensorflow.nn import static_rnn, dynamic_rnn, bidirectional_dynamic_rnn, static_bidirectional_rnn

from makiflow.layers.sf_layer import SimpleForwardLayer
from makiflow.layers.activation_converter import ActivationConverter


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


class RNNLayer(SimpleForwardLayer, ABC):
    def __init__(self, cells, params, named_params_dict, name, seq_length=None, dynamic=True,
                 bidirectional=False):
        self._cells = cells
        self._seq_length = seq_length
        self._dynamic = dynamic
        self._bidirectional = bidirectional
        self._cell_type = CellType.get_cell_type(bidirectional, dynamic)
        super().__init__(name, params, named_params_dict)

    def _forward(self, x):
        if self._cell_type == CellType.BIDIR_DYNAMIC:
            return bidirectional_dynamic_rnn(cell_fw=self._cells, cell_bw=self._cells, inputs=x, dtype=tf.float32)
        elif self._cell_type == CellType.BIDIR_STATIC:
            x = tf.unstack(x, num=self._seq_length, axis=1)
            return static_bidirectional_rnn(cell_fw=self._cells, cell_bw=self._cells, inputs=x, dtype=tf.float32)
        elif self._cell_type == CellType.DYNAMIC:
            return dynamic_rnn(self._cells, x, dtype=tf.float32)
        elif self._cell_type == CellType.STATIC:
            x = tf.unstack(x, num=self._seq_length, axis=1)
            return static_rnn(self._cells, x, dtype=tf.float32)

    def _training_forward(self, x):
        return self._forward(x)

    def get_cells(self):
        return self._cells


class GRULayer(RNNLayer):
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
        cell = GRUCell(num_units=num_cells, activation=activation, dtype=tf.float32)
        cell.build(inputs_shape=[None, tf.Dimension(self._input_dim)])
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

    def to_dict(self):
        return {
            'type': 'GRULayer',
            'params': {
                'num_cells': self._num_cells,
                'input_dim': self._input_dim,
                'seq_length': self._seq_length,
                'name': self._name,
                'dynamic': self._dynamic,
                'bidirectional': self._bidirectional,
                'activation': ActivationConverter.activation_to_str(self._act)
            }
        }


class LSTMLayer(RNNLayer):
    def __init__(self, num_cells, input_dim, seq_length, name, activation=tf.nn.tanh, dynamic=False,
                 bidirectional=False):
        """
        Parameters
        ----------
        num_cells : int
            Number of neurons in the layer.
        input_dim : int
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
        bidirectional : boolean
            Influences whether the layer will be bidirectional.
            WARNING! THIS PARAMETER DOESN'T PLAY ANY ROLE IF YOU'RE GONNA STACK RNN LAYERS.
        """
        self._num_cells = num_cells
        self._input_dim = input_dim
        self._f = activation
        cell = LSTMCell(num_units=num_cells, activation=activation, dtype=tf.float32)
        cell.build(inputs_shape=[None, tf.Dimension(self._input_dim)])
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

    def to_dict(self):
        return {
            'type': 'LSTMLayer',
            'params': {
                'num_cells': self._num_cells,
                'input_dim': self._input_dim,
                'seq_length': self._seq_length,
                'name': self._name,
                'dynamic': self._dynamic,
                'bidirectional': self._bidirectional,
                'activation': ActivationConverter.activation_to_str(self._f)
            }
        }


class RNNBlock(RNNLayer):
    def __init__(self, rnn_layers, seq_length, name, dynamic=False, bidirectional=False, ):
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

    def to_dict(self):
        rnnblock_dict = {
            'type': 'RNNBlock',
            'params': {
                'seq_length': self._seq_length,
                'dynamic': self._dynamic,
                'bidirectional': self._bidirectional,
            }
        }

        rnn_layers_dict = {
            'rnn_layers': []
        }
        for layer in self._rnn_layers:
            rnn_layers_dict['rnn_layers'].append(layer.to_dict())

        rnnblock_dict.update(rnn_layers_dict)
        return rnnblock_dict


class EmbeddingLayer(SimpleForwardLayer):
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

    def _forward(self, x):
        return tf.nn.embedding_lookup(self.embed, x)

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'EmbeddingLayer',
            'params': {
                'num_embeddings': self._num_embeddings,
                'dim': self._dim,
                'name': self._name
            }
        }
