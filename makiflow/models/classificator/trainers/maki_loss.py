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

from ..core import ClassificatorTrainer
from makiflow.core import Loss, TrainerBuilder
import tensorflow as tf


class MakiLossTrainer(ClassificatorTrainer):
    TYPE = 'MakiLossTrainer'
    GAMMA = 'gamma'
    MAKI_LOSS = 'MAKI_LOSS'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: MakiLossTrainer.TYPE,
            TrainerBuilder.PARAMS: {
                MakiLossTrainer.GAMMA: self._maki_gamma
            }
        }

    def set_params(self, params):
        self.set_gamma(params[MakiLossTrainer.GAMMA])

    def _init(self):
        super()._init()
        self._maki_gamma = 1

    def set_gamma(self, gamma):
        """
        Sets the gamma parameter for the Focal Loss. By default the gamma is equal to 1.
        Parameters
        ----------
        gamma : int
            The gamma value.
        """
        assert gamma >= 0, f'Gamma must be non-negative. Received gamma={gamma}'
        assert isinstance(gamma, int), f'Gamma must be an integer. Received gamma={gamma}'
        # noinspection PyAttributeOutsideInit
        self._maki_gamma = gamma

    def _build_loss(self):
        logits = super().get_logits()
        labels = super().get_labels()
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        num_classes = super().get_num_classes()
        logits_shape = tf.shape(logits)
        batch_size = logits_shape[0]

        positives = tf.not_equal(labels, 0)  # [BATCH_SIZE, ...]
        positives_dim_n = len(positives.get_shape())
        axis = list(range(1, positives_dim_n))
        num_positives = tf.reduce_sum(positives, axis=axis)  # [BATCH_SIZE, N_POSITIVES]

        maki_loss = Loss.maki_loss(
            flattened_logits=tf.reshape(logits, shape=[batch_size, -1, num_classes]),
            flattened_labels=tf.reshape(labels, shape=[batch_size, -1]),
            num_classes=num_classes,
            num_positives=num_positives,
            maki_gamma=self._maki_gamma,
            ce_loss=ce_loss
        )
        super().track_loss(maki_loss, MakiLossTrainer.MAKI_LOSS)
        return maki_loss


TrainerBuilder.register_trainer(MakiLossTrainer)
