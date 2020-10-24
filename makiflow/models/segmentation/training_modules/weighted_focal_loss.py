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

from makiflow.core.training.utils import print_train_info, moving_average
from makiflow.core.training.utils import new_optimizer_used, loss_is_built

from ..main_modules import SegmentatorBasic

WEIGHTED_FOCAL_LOSS = 'WEIGHTED FOCAL LOSS'
TOTAL_LOSS = 'TOTAL LOSS'


class WeightedFocalTrainingModule(SegmentatorBasic):
    GAMMA = 'gamma'
    NUM_POSITIVES = 'num_positives'
    WEIGHTED_FOCAL_WEIGHT_MAP = 'weighted_focal_weight_map'

    def _prepare_training_vars(self, use_generator=False):
        self._weighted_focal_loss_is_build = False
        super()._prepare_training_vars(use_generator=use_generator)

    def _build_weighted_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)

        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels

        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self._weighted_focal_gamma)

        flattened_weights = tf.reshape(
            self._weighted_focal_weight_maps, shape=[-1, self.total_predictions]
        )

        num_positives = tf.reduce_sum(self._weighted_focal_num_positives)
        self._weighted_focal_loss = tf.reduce_sum(flattened_weights * focal_weights * self._ce_loss) / num_positives
        self._final_focal_loss = self._build_final_loss(self._weighted_focal_loss)

        self._weighted_focal_loss_is_build = True

    def _setup_weighted_focal_loss_inputs(self):
        self._weighted_focal_gamma = tf.placeholder(tf.float32, shape=[], name=WeightedFocalTrainingModule.GAMMA)

        self._weighted_focal_num_positives = tf.placeholder(tf.float32,
                                                            shape=[self.batch_sz],
                                                            name=WeightedFocalTrainingModule.NUM_POSITIVES
        )

        self._weighted_focal_weight_maps = tf.placeholder(
                                                            tf.float32, shape=[self.batch_sz, self.out_w, self.out_h],
                                                            name=WeightedFocalTrainingModule.WEIGHTED_FOCAL_WEIGHT_MAP
        )

    def _minimize_weighted_focal_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._weighted_focal_loss_is_build:
            self._setup_weighted_focal_loss_inputs()
            self._build_weighted_focal_loss()
            self._weighted_focal_optimizer = optimizer
            self._weighted_focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            loss_is_built()

        if self._weighted_focal_optimizer != optimizer:
            new_optimizer_used()
            self._weighted_focal_optimizer = optimizer
            self._weighted_focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._weighted_focal_train_op

    def fit_weighted_focal(
            self, images, labels, gamma, num_positives, weight_maps, optimizer, epochs=1, global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        gamma : int
            Hyper parameter for FocalLoss.
        num_positives : list
            List of ints. Contains number of `positive samples` per image. `Positive sample` is a pixel
            that is responsible for a class other that background.
        weight_maps : list
            Maps for weighting the loss.
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
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

        train_op = self._minimize_weighted_focal_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        train_total_losses = []
        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                images, labels, num_positives, weight_maps = shuffle(images, labels, num_positives, weight_maps)
                total_loss = 0
                focal_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]

                    NPbatch = num_positives[j * self.batch_sz:(j + 1) * self.batch_sz]
                    WMbatch = weight_maps[j * self.batch_sz:(j + 1) * self.batch_sz]

                    batch_total_loss, batch_focal_loss, _ = self._session.run(
                        [self._final_focal_loss, self._weighted_focal_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._weighted_focal_gamma: gamma,
                            self._weighted_focal_num_positives: NPbatch,
                            self._weighted_focal_weight_maps: WMbatch
                        })
                    # Use exponential decay for calculating loss and error
                    total_loss = moving_average(total_loss, batch_total_loss, j)
                    focal_loss = moving_average(focal_loss, batch_focal_loss, j)

                train_total_losses.append(total_loss)
                train_focal_losses.append(focal_loss)

                print_train_info(i, (WEIGHTED_FOCAL_LOSS, focal_loss), (TOTAL_LOSS, total_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TOTAL_LOSS: train_total_losses,
                WEIGHTED_FOCAL_LOSS: train_focal_losses
            }
