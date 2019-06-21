from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.nn import static_rnn, dynamic_rnn, bidirectional_dynamic_rnn, static_bidirectional_rnn

from makiflow.layers import Layer

class CellType:
    # Bidirectional dynamic
    Bidir_Dynamic = 1
    # Bidirectional static
    Bidir_Static = 2
    # Dynamic
    Dynamic = 3
    # Static
    Static = 4

    @staticmethod
    def get_cell_type(bidirectional, dynamic):
        if bidirectional:
            if dynamic:
                return CellType.Bidir_Dynamic
            else:
                return CellType.Bidir_Static
        else:
            if dynamic:
                return CellType.Dynamic
            else:
                return CellType.Static


class GRULayer(Layer):
    def __init__(self, num_cells, input_dim, seq_length, name, activation=tf.nn.tanh, dynamic=False, bidirectional=False):
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
                WARNING! THIS PARAMETER DOESN'T PLAY ANY ROLE IF YOU'RE GONNA STACK RNN LAYERS.
        """
        self.name = str(name)
        self.num_cells = num_cells
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cells = GRUCell(num_units=num_cells, activation=activation, dtype=tf.float32)
        # Responsible for being RNN whether bidirectional or vanilla
        self.cell_type = CellType.get_cell_type(bidirectional, dynamic)
        self.cells.build(inputs_shape=[None, tf.Dimension(self.input_dim)])
        self.params = self.cells.variables
        self.named_params_dict = { (str(param)+'_'+self.name):param for param in self.params}

    
    def forward(self, X, is_training=False):
        if self.cell_type == CellType.Bidir_Dynamic:
            return bidirectional_dynamic_rnn(cell_fw=self.cells, cell_bw=self.cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Bidir_Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_bidirectional_rnn(cell_fw=self.cells, cell_bw=self.cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Dynamic:
            return dynamic_rnn(self.cells, X, dtype=tf.float32)
        elif self.cell_type == CellType.Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_rnn(self.cells, X, dtype=tf.float32)
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return self.named_params_dict


class LSTMLayer(Layer):
    def __init__(self, num_cells, input_dim, seq_length, name, activation=tf.nn.tanh, dynamic=False, bidirectional=False):
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
                WARNING! THIS PARAMETER DOESN'T PLAY ANY ROLE IF YOU'RE GONNA STACK RNN LAYERS.
        """
        self.name = str(name)
        self.num_cells = num_cells
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cells = LSTMCell(num_units=num_cells, activation=activation, dtype=tf.float32)
        # Responsible for being RNN whether bidirectional or vanilla
        self.cell_type = CellType.get_cell_type(bidirectional, dynamic)
        
        self.cells.build(inputs_shape=[None, tf.Dimension(self.input_dim)])
        self.params = self.cells.variables
        self.named_params_dict = { (str(param)+'_'+self.name):param for param in self.params}

    
    def forward(self, X, is_training=False):
        if self.cell_type == CellType.Bidir_Dynamic:
            return bidirectional_dynamic_rnn(cell_fw=self.cells, cell_bw=self.cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Bidir_Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_bidirectional_rnn(cell_fw=self.cells, cell_bw=self.cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Dynamic:
            return dynamic_rnn(self.cells, X, dtype=tf.float32)
        elif self.cell_type == CellType.Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_rnn(self.cells, X, dtype=tf.float32)
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return self.named_params_dict


class RNNBlock(Layer):
    def __init__(self, rnn_layers, seq_length, name, activation=tf.nn.tanh, dynamic=False, bidirectional=False):
        """
        Parameters
        ----------
            rnn_layers : list
                List of RNN layers to stack.
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
        self.rnn_layers = rnn_layers
        self.rnn_cells = []
        for layer in rnn_layers:
            self.rnn_cells.append(layer.cells)
        self.seq_length = seq_length
        self.stacked_cells = MultiRNNCell(cells=self.rnn_cells)
        self.cell_type = CellType.get_cell_type(bidirectional, dynamic)
        
        self.params = []
        self.named_params_dict = { }
        for layer in rnn_layers:
            self.params += layer.get_params()
            self.named_params_dict.update(layer.get_params_dict())
        


    
    def forward(self, X, is_training=False):
        if self.cell_type == CellType.Bidir_Dynamic:
            return bidirectional_dynamic_rnn(cell_fw=self.stacked_cells, cell_bw=self.stacked_cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Bidir_Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_bidirectional_rnn(cell_fw=self.stacked_cells, cell_bw=self.stacked_cells, inputs=X, dtype=tf.float32)
        elif self.cell_type == CellType.Dynamic:
            return dynamic_rnn(self.stacked_cells, X, dtype=tf.float32)
        elif self.cell_type == CellType.Static:
            X = tf.unstack(X, num=self.seq_length, axis=1)
            return static_rnn(self.stacked_cells, X, dtype=tf.float32)
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return self.named_params_dict
    

class EmbeddingLayer(Layer):
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
                
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.name = 'Embedding_'+str(name)
        embed = np.random.randn(num_embeddings, dim) * np.sqrt(12 / (num_embeddings + dim))
        self.embed = tf.Variable(embed, name=self.name, dtype=tf.float32)
        
        self.params = [self.embed]
        self.named_params_dict = {self.name:self.embed}
    
    def forward(self, X, is_training=False):
        return tf.nn.embedding_lookup(self.embed, X)
    
    def get_params(self):
        return self.params

    def get_params_dict(self):
        return self.named_params_dict
    



    


        