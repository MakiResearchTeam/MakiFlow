import tensorflow as tf

from makiflow.core import Loss, LossFabric
from .single_tensor_loss import SingleTensorLoss


class QCE(SingleTensorLoss):
    def __init__(
            self, tensor_names: list, label_tensors: dict,
            normalize_by_positives=False,
            reduction=Loss.REDUCTION_MEAN, sparse=True
    ):
        """
        Builds Quadratic Cross-Entropy. For reference please see:
        https://ieeexplore.ieee.org/abstract/document/9253199

        Parameters
        ----------
        tensor_names : list
            Contains a single tensor name off of which the loss will be built.
        label_tensors : dict
            Dictionary of tensors that supply label data.
        normalize_by_positives : bool
            Whether to normalize the loss value by the number of positive examples. May improve convergence.
            By default is disabled. When using that normalization, make sure that class 0 is responsible for
            negative examples.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        sparse : bool
            Determines which type of cross-entropy is being computed. Sparse entropy is set
            by default (requires less memory).
        """
        loss_fn = lambda t, lt: QCE.qce(t, lt, reduction, sparse, normalize_by_positives)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def qce(tensors, label_tensors, reduction, sparse, normalize_by_positives):
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

        num_positives = None
        if normalize_by_positives:
            positives = tf.cast(tf.not_equal(labels, 0), tf.float32)  # [BATCH_SIZE, ...]
            positives_dim_n = len(positives.shape())
            axis = list(range(1, positives_dim_n))
            num_positives = tf.reduce_sum(positives, axis=axis)  # [BATCH_SIZE, N_POSITIVES]

        loss = LossFabric.quadratic_ce_loss(ce_loss=loss, num_positives=num_positives, raw_tensor=True)

        if weights:
            loss = loss * weights

        reduction_fn = Loss.REDUCTION_FN[reduction]
        return reduction_fn(loss)
