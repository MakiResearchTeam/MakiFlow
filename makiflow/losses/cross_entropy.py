import tensorflow as tf

from makiflow.core import Loss
from .single_tensor_loss import SingleTensorLoss


class CrossEntropy(SingleTensorLoss):
    def __init__(self, tensor_names: list, label_tensors: dict, reduction=Loss.REDUCTION_MEAN, sparse=True):
        """
        Builds cross-entropy loss.

        Parameters
        ----------
        tensor_names : list
            Contains a single tensor name off of which the loss will be built.
        label_tensors : dict
            Dictionary of tensors that supply label data.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        sparse : bool
            Determines which type of cross-entropy is being computed. Sparse entropy is set
            by default (requires less memory).
        """
        loss_fn = lambda t, lt: CrossEntropy.cross_entropy(t, lt, reduction, sparse)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def cross_entropy(tensors, label_tensors, reduction, sparse):
        preds = tensors[0]
        labels = label_tensors.get(Loss.LABELS)
        weights = label_tensors.get(Loss.WEIGHTS)

        if sparse:
            loss_fn = lambda label, pred: tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=pred
            )
        else:
            loss_fn = lambda label, pred: tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=pred
            )

        loss = loss_fn(labels, preds)

        if weights:
            loss = loss * weights

        reduction_fn = Loss.REDUCTION_FN[reduction]
        return reduction_fn(loss)
