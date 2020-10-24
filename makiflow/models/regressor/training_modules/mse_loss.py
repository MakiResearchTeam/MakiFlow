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
from .additional_losses import BasicTrainingModule
from makiflow.core.training.loss_builder import Loss
from makiflow.core.training.utils import print_train_info, moving_average
from makiflow.core.training.utils import loss_is_built, new_optimizer_used
from sklearn.utils import shuffle
from tqdm import tqdm

MSE_LOSS = 'MSE LOSS'


class MseTrainingModule(BasicTrainingModule):

    def _prepare_training_vars(self):
        self._mse_loss_is_build = False
        super()._prepare_training_vars()

    def _build_mse_loss(self):
        mse_loss = Loss.mse_loss(self._input_x, self._training_out, raw_tensor=True)

        if self._use_weight_mask_for_training:
            mse_loss = tf.reduce_sum(mse_loss * self._weight_mask)
            mse_loss = mse_loss / tf.reduce_sum(self._weight_mask)
        else:
            mse_loss = tf.reduce_mean(mse_loss)

        self._mse_loss = super()._build_additional_losses(mse_loss)
        self._final_mse_loss = self._build_final_loss(self._mse_loss)

    def _setup_mse_loss_inputs(self):
        pass

    def _minimize_mse_loss(self, optimizer, global_step):
        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._mse_loss_is_build:
            self._setup_mse_loss_inputs()
            self._build_mse_loss()
            self._mse_optimizer = optimizer
            self._mse_train_op = optimizer.minimize(
                self._final_mse_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._mse_loss_is_build = True
            loss_is_built()

        if self._mse_optimizer != optimizer:
            new_optimizer_used()
            self._mse_optimizer = optimizer
            self._mse_train_op = optimizer.minimize(
                self._final_mse_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._mse_train_op

    def fit_mse(self, input_x, target_x,
                optimizer, weight_mask=None,
                epochs=1,
                global_step=None, shuffle_data=True):
        """
        Method for training the model.

        Parameters
        ----------
        input_x : list
            Training data
        target_x : list
            Target data
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself
        weight_mask : list
            Weight mask that applies for training. By default equal to None, i. e. not used in training
        epochs : int
            Number of epochs
        global_step
            Please refer to TensorFlow documentation about global step for more info
        shuffle_data : bool
            Set to False if you don't want the data to be shuffled

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_mse_loss(optimizer, global_step)

        n_batches = len(input_x) // self._batch_sz
        mse_losses = []
        iterator = None
        try:
            for i in range(epochs):
                if shuffle_data:
                    if self._use_weight_mask_for_training:
                        input_x, target_x, weight_mask = shuffle(input_x, target_x, weight_mask)
                    else:
                        input_x, target_xs = shuffle(input_x, target_x)
                mse_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = input_x[j * self._batch_sz:(j + 1) * self._batch_sz]
                    Tbatch = target_x[j * self._batch_sz:(j + 1) * self._batch_sz]

                    feed_dict = {
                            self._target_x: Tbatch,
                            self._input_x: Ibatch
                        }

                    if self._use_weight_mask_for_training:
                        Wbatch = weight_mask[j * self._batch_sz:(j + 1) * self._batch_sz]
                        feed_dict[self._weight_mask] = Wbatch

                    batch_mse_loss, _ = self._session.run(
                        [self._final_mse_loss, train_op],
                        feed_dict=feed_dict
                    )

                    # Use exponential decay for calculating loss and error
                    mse_loss = moving_average(mse_loss, batch_mse_loss, j)

                mse_losses.append(mse_loss)

                print_train_info(i, (MSE_LOSS, mse_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MSE_LOSS: mse_losses}

    def gen_fit_mse(self, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model using generator (i. e. pipeline)

        Parameters
        ----------
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself
        epochs : int
            Number of epochs
        iterations : int
            Defines how long one epoch is. One operation is a forward pass using one batch
        global_step
            Please refer to TensorFlow documentation about global step for more info

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_mse_loss(optimizer, global_step)

        mse_losses = []
        iterator = None
        try:
            for i in range(epochs):
                mse_loss = 0
                iterator = tqdm(range(iterations))

                for j in iterator:
                    batch_mse_loss, _ = self._session.run(
                        [self._final_mse_loss, train_op],)
                    # Use exponential decay for calculating loss and error
                    mse_loss = moving_average(mse_loss, batch_mse_loss, j)

                mse_losses.append(mse_loss)

                print_train_info(i, (MSE_LOSS, mse_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MSE_LOSS: mse_losses}
