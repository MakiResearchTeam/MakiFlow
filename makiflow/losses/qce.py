import tensorflow as tf

from makiflow.core import Loss, LossFabric


class QCE(Loss):
    def __init__(
            self, tensor_names, label_tensors: dict,
            normalize_by_positives=False,
            reduction=Loss.REDUCTION_MEAN, sparse=True
    ):
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
