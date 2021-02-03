import tensorflow as tf

from makiflow.core import LossFabric


class MAE(LossFabric):
    LABELS = 'labels'
    WEIGHTS = 'weights'

    def build_loss(self, prediction, label_tensors):
        with tf.name_scope(f'MSE/{self._id}'):
            labels = label_tensors[MAE.LABELS]
            loss = tf.abs(labels - prediction)

            if MAE.WEIGHTS in label_tensors:
                loss = loss * label_tensors[MAE.WEIGHTS]

            loss = tf.reduce_mean(loss)
        return loss
