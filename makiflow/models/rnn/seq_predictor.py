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
import tensorflow as tf
from makiflow.layers import DenseLayer
from sklearn.utils import shuffle
from tqdm import tqdm


class SequencePredictor:
    """
    Used for predicting next word in a sentence.
    """
    def __init__(self, rnn_block, batch_sz, seq_length, embedding, name='MakiSeqPredictor'):
        self.name = name
        self.rnn_block = rnn_block
        self.embedding = embedding
        self.batch_sz = batch_sz
        self.seq_length = seq_length
        self.params = self.rnn_block.get_params()
        

        self.X = tf.placeholder(tf.float32, shape=(batch_sz, seq_length), name='input')
        self.Y = tf.placeholder(tf.float32, shape=(batch_sz, seq_length), name='targets')

    
    def __setup_params(self):
        self.params = self.rnn_block.get_params()
        self.named_params_dict = self.rnn_block.get_params_dict()
        # The dimensionality of the RNN output will be the dimensionality
        # of the output of the last RNN layer in the RNN block, i.e. number
        # of the RNN cells
        self.embedding_size = self.embedding.num_embeddings
        self.last_out_dim = self.rnn_block.rnn_layers[-1].num_cells
        self.classification_layer = DenseLayer(
            in_d=self.last_out_dim,
            out_d=self.embedding_size,
            name='classification_Layer',
            activation=None)
        self.params += [self.embedding]
        self.params += self.classification_layer.get_params()
        self.named_params_dict.update(self.embedding.get_params_dict())
        self.named_params_dict.update(self.classification_layer.get_params_dict())

    
    def set_session(self, session):
        assert(session is not None)
        self.session = session
        init_op = tf.variables_initializer(self.params)
        session.run(init_op)


    def __forward_train(self, X):
        X = self.embedding.forward(X)
        # States are the states of the last RNN layer in the block
        # It is a tensor of shape [batch_sz, seq_length, num_cells]
        # Outs are the outs of the each RNN layer in the block
        # It is a tensor of shape [num_layers, batch_sz, num_cells_per_layer]
        states, outs = self.rnn_block.forward(X)
        flatten_states = tf.reshape(states, (self.batch_sz*self.seq_length, self.last_out_dim))
        # We don't use classification layer yet cause we're gonna use sampled softmax loss,
        # i.e. we're gonna use classification layer in the fit method
        return flatten_states
    

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, num_loss_samples=50, epochs=1, test_period=1):
        nce_weights = tf.transpose(self.classification_layer.W, (1, 0))
        nce_biases = self.classification_layer.b

        states = self.__forward_train(self.X)
        labels = tf.reshape(self.Y, (self.batch_sz*self.seq_length, 1))
        cost = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=states,
                num_sampled=num_loss_samples,
                num_classes=self.embedding_size
            )
        )

        train_op = optimizer.minimize(cost)
        # Initialize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables()))
        n_batches = len(Xtrain) // self.batch_sz
        for i in range(epochs):
            cost_val = 0
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in tqdm(range(n_batches)):
                Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]

                _, c = self.session.run(
                    [train_op, cost],
                    feed_dict={
                        self.X: Xbatch,
                        self.Y: Ybatch
                    }
                )
                cost_val += c
            print('epoch', i,'cost', cost_val)

        

    
    

    def predict(self, X):
        pass

        
