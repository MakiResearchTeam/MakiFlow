import tensorflow as tf

from makiflow.core import LossFabric


class MSE(LossFabric):
    LABELS = 'labels'
    WEIGHTS = 'weights'

    def build_loss(self, prediction, label_tensors):
        with tf.name_scope(f'MSE/{self._id}'):
            labels = label_tensors[MSE.LABELS]
            loss = tf.square(labels - prediction)

            if MSE.WEIGHTS in label_tensors:
                loss = loss * label_tensors[MSE.WEIGHTS]

            loss = tf.reduce_mean(loss)
        return loss
