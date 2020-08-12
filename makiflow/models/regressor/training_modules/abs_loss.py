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
from makiflow.base.loss_builder import Loss
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
from tqdm import tqdm
from sklearn.utils import shuffle

ABS_LOSS = 'ABS LOSS'


class AbsTrainingModule(BasicTrainingModule):

    def _prepare_training_vars(self):
        self._abs_loss_is_build = False
        super()._prepare_training_vars()

    def _build_abs_loss(self):
        abs_loss = Loss.abs_loss(self._input_x, self._training_out, raw_tensor=True)

        if self._use_weight_mask_for_training:
            abs_loss = tf.reduce_sum(abs_loss * self._weight_mask)
            abs_loss = abs_loss / tf.reduce_sum(self._weight_mask)
        else:
            abs_loss = tf.reduce_mean(abs_loss)

        self._abs_loss = super()._build_additional_losses(abs_loss)
        self._final_abs_loss = self._build_final_loss(self._abs_loss)

    def _setup_abs_loss_inputs(self):
        pass

    def _minimize_abs_loss(self, optimizer, global_step):
        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._abs_loss_is_build:
            self._setup_abs_loss_inputs()
            self._build_abs_loss()
            self._abs_optimizer = optimizer
            self._abs_train_op = optimizer.minimize(
                self._final_abs_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._abs_loss_is_build = True
            loss_is_built()

        if self._abs_optimizer != optimizer:
            new_optimizer_used()
            self._abs_optimizer = optimizer
            self._abs_train_op = optimizer.minimize(
                self._final_abs_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._abs_train_op

    def fit_abs(self, input_x, target_x, optimizer, weight_mask=None, epochs=1, global_step=None, shuffle_data=True):
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

        train_op = self._minimize_abs_loss(optimizer, global_step)

        n_batches = len(input_x) // self._batch_sz
        train_abs_losses = []
        iterator = None
        try:
            for i in range(epochs):
                if shuffle_data:
                    if self._use_weight_mask_for_training:
                        input_x, target_x, weight_mask = shuffle(input_x, target_x, weight_mask)
                    else:
                        input_x, target_x = shuffle(input_x, target_x)
                abs_loss = 0
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

                    batch_abs_loss, _ = self._session.run(
                        [self._final_abs_loss, train_op],
                        feed_dict=feed_dict
                    )

                    # Use exponential decay for calculating loss and error
                    abs_loss = moving_average(abs_loss, batch_abs_loss, j)

                train_abs_losses.append(abs_loss)

                print_train_info(i, (ABS_LOSS, abs_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {ABS_LOSS: train_abs_losses}

    def gen_fit_abs(self, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model using generator (i. e. pipeline)

        Parameters
        ----------
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself
        epochs : int
            Number of epochs
        iterations : int
            Defines how long one epoch is. One operation is a forward pass
            using one batch.
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

        train_op = self._minimize_abs_loss(optimizer, global_step)

        abs_losses = []
        iterator = None
        try:
            for i in range(epochs):
                abs_loss = 0
                iterator = tqdm(range(iterations))

                for j in iterator:
                    batch_abs_loss, _ = self._session.run(
                        [self._final_abs_loss, train_op],)
                    # Use exponential decay for calculating loss and error
                    abs_loss = moving_average(abs_loss, batch_abs_loss, j)

                abs_losses.append(abs_loss)

                print_train_info(i, (ABS_LOSS, abs_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {ABS_LOSS: abs_losses}

