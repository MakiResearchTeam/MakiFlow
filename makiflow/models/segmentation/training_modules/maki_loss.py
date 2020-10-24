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

from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

from scipy.special import binom

from makiflow.generators.segmentator import SegmentIterator
from makiflow.core.training.common import print_train_info, moving_average
from makiflow.core.training.common import new_optimizer_used, loss_is_built

from ..main_modules import SegmentatorBasic

MAKI_LOSS = 'MAKI LOSS'


class MakiTrainingModule(SegmentatorBasic):
    NUM_POSITIVES = 'num_positives'

    def _prepare_training_vars(self, use_generator=False):
        self._maki_loss_is_build = False
        super()._prepare_training_vars(use_generator=use_generator)

    def _create_maki_polynom_part(self, k, sparse_confidences):
        binomial_coeff = binom(self._maki_gamma, k)
        powered_p = tf.pow(-1.0 * sparse_confidences, k)
        return binomial_coeff * powered_p / (1.0 * k)

    def _build_maki_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)

        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels

        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)

        # Create Maki polynomial
        maki_polynomial = tf.constant(0.0)
        for k in range(1, self._maki_gamma + 1):
            # Do subtraction because gradient must be with minus as well
            # Maki loss grad: -(1 - p)^gamma / p
            # CE loss grad: - 1 / p
            maki_polynomial -= self._create_maki_polynom_part(k, sparse_confidences) - \
                               self._create_maki_polynom_part(k, tf.ones_like(sparse_confidences))

        num_positives = tf.reduce_sum(self._maki_num_positives)
        self._maki_loss = tf.reduce_sum(maki_polynomial + self._ce_loss) / num_positives
        self._final_maki_loss = self._build_final_loss(self._maki_loss)
        self._maki_loss_is_build = True

    def _setup_maki_loss_inputs(self):
        self._maki_gamma = None
        if self._use_generator:
            self._maki_num_positives = self._generator.get_iterator()[SegmentIterator.NUM_POSITIVES]
        else:
            self._maki_num_positives = tf.placeholder(tf.float32,
                                                      shape=[self.batch_sz],
                                                      name=MakiTrainingModule.NUM_POSITIVES
            )

    def _minimize_maki_loss(self, optimizer, global_step, gamma):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._maki_loss_is_build:
            self._setup_maki_loss_inputs()
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            loss_is_built()

        if self._maki_gamma != gamma:
            print('New gamma is used.')
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_optimizer != optimizer:
            new_optimizer_used()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._maki_train_op

    def fit_maki(self, images, labels, gamma: int, num_positives, optimizer, epochs=1, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        gamma : int
            Hyper parameter for MakiLoss.
        num_positives : list
            List of ints. Contains number of `positive samples` per image. `Positive sample` is a pixel
            that is responsible for a class other that background.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert optimizer is not None, "Optimizer can't be None, please set it up."
        assert self._session is not None, "Session has not been set, please use set it up with `set_session` method."
        assert type(gamma) == int, "The number of gamma must be an integer."

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        n_batches = len(images) // self.batch_sz
        train_maki_losses = []
        iterator = None
        try:
            for i in range(epochs):
                images, labels = shuffle(images, labels)
                maki_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    NPbatch = num_positives[j * self.batch_sz:(j + 1) * self.batch_sz]

                    batch_maki_loss, _ = self._session.run(
                        [self._final_maki_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._maki_num_positives: NPbatch
                        })
                    # Use exponential decay for calculating loss and error
                    maki_loss = moving_average(maki_loss, batch_maki_loss, j)

                train_maki_losses.append(maki_loss)
                print_train_info(i, (MAKI_LOSS, maki_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MAKI_LOSS: train_maki_losses}

    def genfit_maki(self, gamma: int, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        gamma : int
            Hyper parameter for MakiLoss.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        iterations : int
            Defines how ones epoch is. One operation is a forward pass
            using one batch.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert optimizer is not None, "Optimizer can't be None, please set it up."
        assert self._session is not None, "Session has not been set, please use set it up with `set_session` method."
        assert type(gamma) == int, "The number of gamma must be an integer."

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        train_maki_losses = []
        iterator = None
        try:
            for i in range(epochs):
                maki_loss = 0
                iterator = tqdm(range(iterations))

                for j in iterator:
                    batch_maki_loss, _ = self._session.run(
                        [self._final_maki_loss, train_op]
                    )
                    # Use exponential decay for calculating loss and error
                    maki_loss = moving_average(maki_loss, batch_maki_loss, j)

                train_maki_losses.append(maki_loss)

                print_train_info(i, (MAKI_LOSS, maki_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MAKI_LOSS: train_maki_losses}

