import tensorflow as tf

from makiflow.core import Loss


class CrossEntropy(Loss):
    def __init__(self, tensor_names, label_tensors: dict, reduction=Loss.REDUCTION_MEAN, sparse=True):
        loss_fn = lambda t, lt: CrossEntropy.cross_entropy(t, lt, reduction, sparse)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def cross_entropy(tensors, label_tensors, reduction, sparse):
        preds = tensors[0]
        labels = label_tensors.get(CrossEntropy.LABELS)
        weights = label_tensors.get(CrossEntropy.WEIGHTS)

        if sparse:
            loss_fn = lambda label, pred: tf.nn.sparse_softmax_cross_entropy_with_logits(
                label=label, logits=pred
            )
        else:
            loss_fn = lambda label, pred: tf.nn.softmax_cross_entropy_with_logits(
                label=label, logits=pred
            )

        loss = loss_fn(labels, preds)

        if weights:
            loss = loss * weights

        reduction_fn = CrossEntropy.REDUCTION_FN[reduction]
        return reduction_fn(loss)
