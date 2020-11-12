from ..core import SSPTrainer
from makiflow.core.training import Loss
import tensorflow as tf


class CETrainer(SSPTrainer):
    def _build_head_losses(self, coords, point_indicators, human_indicators, label_coords, label_point_indicators,
                           label_human_indicators) -> tuple:
        n_positives = tf.reduce_sum(label_human_indicators)

        with tf.name_scope(SSPTrainer.COORDS_LOSS):
            coords_loss = Loss.mse_loss(
                labels=label_coords, predictions=coords, raw_tensor=True
            )
            b, h, w, c = coords_loss.get_shape().as_list()
            coords_loss = tf.reshape(coords_loss, shape=[b, h, w, c // 2, 2])
            # Mask out the loss of the absent points
            coords_loss = coords_loss * tf.expand_dims(point_indicators, axis=-1)
            coords_loss = tf.reshape(coords_loss, shape=[b, h, w, c]) * human_indicators
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
