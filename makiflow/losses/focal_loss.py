import tensorflow as tf

from makiflow.core import Loss, LossFabric


class FocalLoss(Loss):
    def __init__(
            self, tensor_names, label_tensors: dict,
            num_classes, gamma=2.0, normalize_by_positives=False,
            reduction=Loss.REDUCTION_MEAN
    ):
        loss_fn = lambda t, lt: FocalLoss.focal_loss(t, lt, reduction, num_classes, gamma, normalize_by_positives)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def focal_loss(tensors, label_tensors, reduction, num_classes, gamma, normalize_by_positives):
        logits = tensors[0]
        labels = label_tensors.get(Loss.LABELS)
        weights = label_tensors.get(Loss.WEIGHTS)

        num_positives = None
        if normalize_by_positives:
            positives = tf.cast(tf.not_equal(labels, 0), tf.float32)  # [BATCH_SIZE, ...]
            positives_dim_n = len(positives.shape())
            axis = list(range(1, positives_dim_n))
            num_positives = tf.reduce_sum(positives, axis=axis)  # [BATCH_SIZE, N_POSITIVES]

        loss = LossFabric.focal_loss(
            logits=logits,
            labels=labels,
            num_classes=num_classes,
            num_positives=num_positives,
            focal_gamma=gamma,
            raw_tensor=True
        )

        if weights:
            loss = loss * weights

        reduction_fn = Loss.REDUCTION_FN[reduction]
        return reduction_fn(loss)
