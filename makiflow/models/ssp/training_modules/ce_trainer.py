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

from ..core import SSPTrainer
from makiflow.core.training import Loss
import tensorflow as tf


class CETrainer(SSPTrainer):
    def _build_head_losses(
            self,
            coords, point_indicators, human_indicators,
            label_coords, label_point_indicators, label_human_indicators
    ) -> tuple:
        n_positives = tf.maximum(tf.reduce_sum(label_human_indicators), 1.0)

        with tf.name_scope(SSPTrainer.COORDS_LOSS):
            coords_loss = Loss.mse_loss(
                labels=label_coords, predictions=coords, raw_tensor=True
            )
            b, h, w, c = coords_loss.get_shape().as_list()
            coords_loss = tf.reshape(coords_loss, shape=[b, h, w, c // 2, 2])
            # Mask out the loss of the absent points
            coords_loss = coords_loss * tf.expand_dims(label_point_indicators, axis=-1)
            coords_loss = tf.reshape(coords_loss, shape=[b, h, w, c]) * label_human_indicators
            coords_loss = tf.reduce_mean(coords_loss, axis=-1)
            coords_loss = tf.reduce_sum(coords_loss) / n_positives

        with tf.name_scope(SSPTrainer.POINT_INDICATORS_LOSS):
            point_indicators_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=point_indicators, labels=label_point_indicators
            ) * label_human_indicators
            point_indicators_loss = tf.reduce_mean(point_indicators_loss, axis=-1)
            point_indicators_loss = tf.reduce_sum(point_indicators_loss) / n_positives

        with tf.name_scope(SSPTrainer.HUMAN_INDICATORS_LOSS):
            human_indicators_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=human_indicators, labels=label_human_indicators
            )
            human_indicators_loss = tf.reduce_mean(human_indicators_loss)

        return coords_loss, point_indicators_loss, human_indicators_loss


if __name__ == '__main__':
    from ..core.debug_utils import ssp_model, InputLayer

    model = ssp_model()
    trainer = CETrainer(model, train_inputs=[InputLayer(input_shape=[1, 64, 64, 3], name='input_image')])
    trainer.compile()
