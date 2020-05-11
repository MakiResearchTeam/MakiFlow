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

from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built

from ..main_modules import SegmentatorBasic

QUADRATIC_CE_LOSS = 'QUADRATIC CE LOSS'
TOTAL_LOSS = 'TOTAL LOSS'


class QuadraticCrossEntropyTrainingModule(SegmentatorBasic):

    def _prepare_training_vars(self, use_generator=False):
        self._quadratic_ce_loss_is_build = False
        super()._prepare_training_vars(use_generator=use_generator)

    def _build_quadratic_ce_loss(self):
        # [batch_sz, total_predictions, num_classes]
        quadratic_ce = self._ce_loss * self._ce_loss / 2.0

        self._quadratic_ce = tf.reduce_mean(quadratic_ce)
        self._final_quadratic_ce_loss = self._build_final_loss(self._quadratic_ce)
        self._quadratic_ce_loss_is_build = True

    def _setup_quadratic_ce_loss_inputs(self):
        pass

    def _minimize_quadratic_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._quadratic_ce_loss_is_build:
            self._setup_quadratic_ce_loss_inputs()
            self._build_quadratic_ce_loss()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            loss_is_built()

        if self._quadratic_ce_optimizer != optimizer:
            new_optimizer_used()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._quadratic_ce_train_op

    def fit_quadratic_ce(
            self, images, labels, optimizer, epochs=1, global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
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

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        iterator = None
        train_total_losses = []
        train_quadratic_ce_losses = []
        try:
            for i in range(epochs):
                images, labels = shuffle(images, labels)
                total_loss = 0
                quadratic_ce_loss = 0
                iterator = tqdm(range(n_batches))
                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_quadratic_ce_loss, batch_total_loss, _ = self._session.run(
                        [self._final_quadratic_ce_loss, self._quadratic_ce, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch
                        }
                    )
                    # Use exponential decay for calculating loss and error
                    total_loss = moving_average(total_loss, batch_total_loss, j)
                    quadratic_ce_loss = moving_average(quadratic_ce_loss, batch_quadratic_ce_loss, j)

                train_total_losses.append(total_loss)
                train_quadratic_ce_losses.append(quadratic_ce_loss)

                print_train_info(i, (TOTAL_LOSS, total_loss), (QUADRATIC_CE_LOSS, quadratic_ce_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TOTAL_LOSS: train_total_losses,
                QUADRATIC_CE_LOSS: train_quadratic_ce_losses
            }

    def genfit_quadratic_ce(
            self, optimizer, epochs=1, iterations=10, global_step=None
    ):
        """
        Method for training the model using generator.

        Parameters
        ----------
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

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        iterator = None
        train_total_losses = []
        train_quadratic_ce_losses = []
        try:
            for i in range(epochs):
                total_loss = 0
                quadratic_ce_loss = 0
                iterator = tqdm(range(iterations))
                for j in iterator:
                    batch_quadratic_ce_loss, batch_total_loss, _ = self._session.run(
                        [self._final_quadratic_ce_loss, self._quadratic_ce, train_op]
                    )
                    # Use exponential decay for calculating loss and error
                    total_loss = moving_average(total_loss, batch_total_loss, j)
                    quadratic_ce_loss = moving_average(quadratic_ce_loss, batch_quadratic_ce_loss, j)

                train_total_losses.append(total_loss)
                train_quadratic_ce_losses.append(quadratic_ce_loss)
                print_train_info(i, (TOTAL_LOSS, total_loss), (QUADRATIC_CE_LOSS, quadratic_ce_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TOTAL_LOSS: train_total_losses,
                QUADRATIC_CE_LOSS: train_quadratic_ce_losses
            }

