from ..core import SSPTrainer
from makiflow.core.training import Loss
import tensorflow as tf


class CETrainer(SSPTrainer):
    def _build_head_losses(self, coords, point_indicators, human_indicators, label_coords, label_point_indicators,
                           label_human_indicators) -> tuple:

        n_positives = tf.reduce_sum(label_human_indicators)
        coords_loss = Loss.mse_loss(labels=label_coords, predictions=coords, raw_tensor=True) * label_human_indicators
        coords_loss = tf.reduce_sum(coords_loss) / n_positives

        point_indicators_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=point_indicators, labels=label_point_indicators
        ) * label_human_indicators
        point_indicators_loss = tf.reduce_sum(point_indicators_loss) / n_positives

        human_indicators_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=human_indicators, labels=label_human_indicators
        )
        human_indicators_loss = tf.reduce_sum(human_indicators_loss) / n_positives

        return coords_loss, point_indicators_loss, human_indicators_loss


if __name__ == '__main__':
    from ..core.debug_utils import ssp_model, InputLayer

    model = ssp_model()
    trainer = CETrainer(model, train_inputs=[InputLayer(input_shape=[1, 64, 64, 3], name='input_image')])
    trainer.compile()
