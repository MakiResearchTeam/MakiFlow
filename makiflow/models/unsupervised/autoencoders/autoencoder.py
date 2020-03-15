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
from makiflow.models.unsupervised.autoencoders.encoder import Encoder
from makiflow.models.unsupervised.autoencoders import Decoder

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

class AutoEncoder:
    def __init__(self, encoder: Encoder, decoder: Decoder, name):
        """
        Parameters
        ----------
        encoder : Encoder
            pass
        decoder : Decoder
            pass
        name : str
            Name of the model.
        """
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = self.encoder.input_shape
        self.batch_sz = self.input_shape[0]

        # Collect all the params for later initialization
        self.params = []
        self.params += encoder.get_params()
        self.params += decoder.get_params()


        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        self.named_params_dict.update(encoder.get_named_params())
        self.named_params_dict.update(decoder.get_named_params())


        self.X = tf.placeholder(tf.float32, shape=self.input_shape)

        # For training
        self.encoder_out = encoder.forward(self.X)
        self.decoder_out = decoder.forward(self.encoder_out)

        # For testing
        self.encoder_out_test = encoder.forward(self.X,is_training=False)
        self.decoder_out_test = decoder.forward(self.encoder_out_test,is_training=False)

    
    def encode(self, X):
        assert(self.session is not None)
        n_batches = X.shape[0] // self.batch_sz
        result = []
        for i in tqdm(range(n_batches)):
            Xbatch = X[i*self.batch_sz:(i+1)*self.batch_sz]
            result.append(
                self.session.run(
                    self.encoder_out_test,
                    feed_dict={self.X: Xbatch}
                )
            )
        return np.vstack(result)

    
    def set_session(self, session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)
        self.encoder.session = session
        self.decoder.session = session

    
    def save_weights(self, path):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        save_path = saver.save(self.session, path)
        print('Model saved to %s' % save_path)

    
    def load_weights(self, path):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        saver.restore(self.session, path)
        print('Model restored')


    def fit(self, Xtrain, Xtest, optimizer=None, epochs=1, test_period=1):
        """
        Method for training the model.

        Parameters
        ----------
            Xtrain : numpy array
                Training samples.
            Xtest : numpy array
                Testing samples.
            optimizer : tensorflow optimizer
                Model uses tensorflow optimizer in order to train itself.
            epochs : int
                Number of training epochs.
            test_period : int
                Test begins each `test_period` epochs. You can set a larger number in order to
                speed up training.
        
        Returns
        -------
            python dictionary
                Dictionary with all testing data: train and test losses.
        """
        assert (optimizer is not None)
        assert (self.session is not None)

        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        
        loss = tf.reduce_mean( tf.losses.mean_squared_error(self.X, self.decoder_out) )
        train_op = (loss, optimizer.minimize(loss))
        # Initilize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables()))

        # For testing
        test_loss = tf.reduce_mean( tf.losses.mean_squared_error(self.X, self.decoder_out_test) )

        n_batches = Xtrain.shape[0] // self.batch_sz

        train_losses = []
        test_losses = []
        for i in range(epochs):
            Xtrain = shuffle(Xtrain)
            train_loss = 0
            iterator = range(n_batches)
            
            for j in tqdm(iterator):
                Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]

                train_loss_batch, _ = self.session.run(
                    train_op,
                    feed_dict={self.X: Xbatch})
                # Use exponential decay for calculating loss
                train_loss = 0.99 * train_loss + 0.01 * train_loss_batch
        
            # Validation the network on test data
            if (i % test_period) == 0:
                test_loss_value = 0
                test_n_batches = len(Xtest) // self.batch_sz
                for j in range(test_n_batches):
                    Xtestbatch = Xtest[j*self.batch_sz:(j+1)*self.batch_sz]
                    test_loss_value += self.session.run(
                        test_loss,
                        feed_dict={self.X: Xtestbatch})

                # Collect and print data
                test_loss_value = test_loss_value / test_n_batches

                train_losses.append(train_loss)
                test_losses.append(test_loss_value)
                print('Epoch:', i, 'Train cost: {:0.5f}'.format(train_loss),
                    'Test cost: {:0.5f}'.format(test_loss_value))

        return {'train losses': train_losses, 'test losess': test_losses}
                

