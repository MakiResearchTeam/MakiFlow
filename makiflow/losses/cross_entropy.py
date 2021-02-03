import tensorflow as tf

from makiflow.core import Loss


class CrossEntropy(Loss):
    LABELS = 'labels'
    WEIGHTS = 'weights'

    def build_loss(self, prediction, label_tensors):
        with tf.name_scope(f'CrossEntropy/{self._id}'):
            labels = label_tensors[CrossEntropy.LABELS]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=prediction
            )

            if CrossEntropy.WEIGHTS in label_tensors:
                loss = loss * label_tensors[CrossEntropy.WEIGHTS]

            loss = tf.reduce_mean(loss)
        return loss
