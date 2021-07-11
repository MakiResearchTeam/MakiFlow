import tensorflow as tf

from makiflow.core import Loss, LossFabric
from .single_tensor_loss import SingleTensorLoss


class FocalLoss(SingleTensorLoss):
    def __init__(
            self, tensor_names: list, label_tensors: dict,
            num_classes, gamma=2.0, normalize_by_positives=False,
            reduction=Loss.REDUCTION_MEAN
    ):
        """
        Builds Focal loss. For reference please see:
        https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf

        Parameters
        ----------
        tensor_names : list
            Contains a single tensor name off of which the loss will be built.
        label_tensors : dict
            Dictionary of tensors that supply label data.
        num_classes : int
            Number of classes.
        gamma : float
            The focal loss hyperparameter. Determines how much penalty well-classified examples receive:
            larger the gamma, lesser the penalty. Depends on the balance of the classes (for severe imbalance
            a larger gamma is advised).
        normalize_by_positives : bool
            Whether to normalize the loss value by the number of positive examples. May improve convergence.
            By default is disabled. When using that normalization, make sure that class 0 is responsible for
            negative examples.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        """
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
