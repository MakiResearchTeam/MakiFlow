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


class FocalLossWweightsMaskAsLastMaskTrainer(ClassificatorTrainer):
    TYPE = 'FocalLossWweightsMaskAsLastMaskTrainer'
    GAMMA = 'gamma'
    NORM_BY_POS = 'norm_by_pos'
    INDX_MASK_LAST = 'indx_mask_last'

    FOCAL_LOSS = 'FOCAL_LOSS'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: FocalLossWweightsMaskAsLastMaskTrainer.TYPE,
            TrainerBuilder.PARAMS: {
                FocalLossWweightsMaskAsLastMaskTrainer.GAMMA: self._focal_gamma
            }
        }

    def set_params(self, params):
        self.set_gamma(params[FocalLossWweightsMaskAsLastMaskTrainer.GAMMA])
        self.set_norm_by_pos(params[FocalLossWweightsMaskAsLastMaskTrainer.NORM_BY_POS])
        self.set_indx_weight_mask_last(params[FocalLossWweightsMaskAsLastMaskTrainer.INDX_MASK_LAST])

    def _init(self):
        super()._init()
        self._focal_gamma = 2.0
        self._normalize_by_positives = False
        self._indx_weight_mask = 0

    def set_indx_weight_mask_last(self, indx_weight_mask: int):
        assert indx_weight_mask > 0, f"Weight inx of mask must be grater than 0, but {indx_weight_mask} was given"
        self._indx_weight_mask = indx_weight_mask

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
        num_classes = super().get_num_classes()

        w_mask = tf.cast(labels != self._indx_weight_mask, dtype=tf.int32)
        zero_region = tf.cast(labels != 0, dtype=tf.int32)
        dont_care_and_background_reg = w_mask * zero_region
        num_w_mask_elem = tf.cast(tf.equal(dont_care_and_background_reg, 0), tf.float32)

        num_positives = None
        if self._normalize_by_positives:
            positives = tf.cast(tf.not_equal(dont_care_and_background_reg, 0), tf.float32) # [BATCH_SIZE, ...]
            positives_dim_n = len(positives.get_shape())
            axis = list(range(1, positives_dim_n))
            num_positives = tf.reduce_sum(positives, axis=axis)  # [BATCH_SIZE, N_POSITIVES]
        labels = labels * dont_care_and_background_reg # Zero masked zones as background
        focal_loss = Loss.focal_loss(
            logits=logits,
            labels=labels,
            num_classes=num_classes,
            num_positives=num_positives,
            focal_gamma=self._focal_gamma,
            raw_tensor=True
        )
        focal_loss = tf.reduce_sum(focal_loss * tf.cast(w_mask, dtype=tf.float32)) # Apply weight on certain regions

        if not self._normalize_by_positives:
            focal_loss = focal_loss / float(super().get_batch_size())

        super().track_loss(focal_loss, FocalLossWweightsMaskAsLastMaskTrainer.FOCAL_LOSS)
        return focal_loss


TrainerBuilder.register_trainer(FocalLossWweightsMaskAsLastMaskTrainer)
