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


class FocalBinaryTrainer(ClassificatorTrainer):
    TYPE = 'FocalBinaryTrainer'
    GAMMA = 'gamma'
    NORM_BY_POS = 'norm_by_pos'

    FOCAL_LOSS = 'FOCAL_BINARY_LOSS'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: FocalBinaryTrainer.TYPE,
            TrainerBuilder.PARAMS: {
                FocalBinaryTrainer.GAMMA: self._focal_gamma
            }
        }

    def set_params(self, params):
        self.set_gamma(params[FocalBinaryTrainer.GAMMA])
        self.set_norm_by_pos(params[FocalBinaryTrainer.NORM_BY_POS])
        super().set_params(params)

    def _init(self):
        super()._init()
        self._focal_gamma = 2.0
        self._normalize_by_positives = False

    def set_gamma(self, gamma):
        """
        Sets the gamma parameter for the Focal Loss. By default the gamma is equal to 2.0.
        Parameters
        ----------
        gamma : float
            The gamma value.
        """
        assert gamma >= 0, f'Gamma must be non-negative. Received gamma={gamma}'
        # noinspection PyAttributeOutsideInit
        self._focal_gamma = gamma

    def normalize_by_positives(self):
        """
        Enables loss normalization by the number of positive samples in the batch.
        """
        self._normalize_by_positives = True

    def set_norm_by_pos(self, norm_by_pos: bool):
        self._normalize_by_positives = norm_by_pos

    def _build_loss(self):
        logits = super().get_logits()
        labels = super().get_labels()

        num_positives = None
        if self._normalize_by_positives:
            positives = tf.cast(tf.not_equal(labels, 0), tf.float32)  # [BATCH_SIZE, ...]
            positives_dim_n = len(positives.get_shape())
            axis = list(range(1, positives_dim_n))
            num_positives = tf.reduce_sum(positives, axis=axis)  # [BATCH_SIZE, N_POSITIVES]

        focal_loss = Loss.focal_binary_loss(
            logits=logits,
            labels=labels,
            num_positives=num_positives,
            focal_gamma=self._focal_gamma,
            label_smoothing=self._smoothing_labels
        )

        if not self._normalize_by_positives:
            focal_loss = focal_loss / float(super().get_batch_size())

        super().track_loss(focal_loss, FocalBinaryTrainer.FOCAL_LOSS)
        return focal_loss


TrainerBuilder.register_trainer(FocalBinaryTrainer)
