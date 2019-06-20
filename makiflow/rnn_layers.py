from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.nn import static_rnn, dynamic_rnn, bidirectional_dynamic_rnn, static_bidirectional_rnn

from makiflow.layers import Layer


class GRULayer(Layer):
    def __init__(self, num_cells, input_dim, seq_length, activation=tf.nn.tanh, dynamic=False, bidirectional=False):
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
            bidirectional : boolean
                Influences whether the layer will be bidirectional.
        """

        self.num_cells = num_cells
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cells = GRUCell(num_units=num_cells, activation=activation)
        self.params = self.cells.trainable_variables()
        # Responsible for being RNN whether bidirectional or vanilla
        if bidirectional:
            if dynamic:
                self.wrap = bidirectional_dynamic_rnn
            else:
                self.wrap = static_bidirectional_rnn
        else:
            if dynamic:
                self.wrap = dynamic_rnn
            else:
                self.wrap = static_rnn
        
        self.cells.build(inputs_shape=[None, tf.Dimension(self.input_dim)])
        self.params = self.cells.variables
        self.named_params_dict = { str(param):param for param in self.params}

    
    def forward(self, X, is_training=False):
        if dynamic_rnn:
            # X of shape [batch_sz, max_time, ...]
            return self.wrap(cell=self.cells, inputs=X)
        else:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            # X now is a list of shape [max_time, batch_sz, ...]
            return self.wrap(cell=self.cells, inputs=X)
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return self.named_params_dict


class LSTMLayer(Layer):
    def __init__(self, num_cells, input_dim, seq_length, activation=tf.nn.tanh, dynamic=False, bidirectional=False):
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
            bidirectional : boolean
                Influences whether the layer will be bidirectional.
        """

        self.num_cells = num_cells
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cells = LSTMCell(num_units=num_cells, activation=activation)
        self.params = self.cells.trainable_variables()
        # Responsible for being RNN whether bidirectional or vanilla
        if bidirectional:
            if dynamic:
                self.wrap = bidirectional_dynamic_rnn
            else:
                self.wrap = static_bidirectional_rnn
        else:
            if dynamic:
                self.wrap = dynamic_rnn
            else:
                self.wrap = static_rnn
        
        self.cells.build(inputs_shape=[None, tf.Dimension(self.input_dim)])
        self.params = self.cells.variables
        self.named_params_dict = { str(param):param for param in self.params}

    
    def forward(self, X, is_training=False):
        if dynamic_rnn:
            # X of shape [batch_sz, max_time, ...]
            return self.wrap(cell=self.cells, inputs=X)
        else:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            # X now is a list of shape [max_time, batch_sz, ...]
            return self.wrap(cell=self.cells, inputs=X)
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return self.named_params_dict

    
    


        