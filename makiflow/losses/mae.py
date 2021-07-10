import tensorflow as tf

from makiflow.core import Loss
from .single_tensor_loss import SingleTensorLoss


class MAE(SingleTensorLoss):
    def __init__(self, tensor_names, label_tensors: dict, reduction=Loss.REDUCTION_MEAN):
        loss_fn = lambda t, lt: MAE.mean_absolute_error(t, lt, reduction)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def mean_absolute_error(tensors, label_tensors, reduction):
        preds = tensors[0]
        labels = label_tensors.get(Loss.LABELS)
        weights = label_tensors.get(Loss.WEIGHTS)

        loss = tf.abs(labels - preds)

        if weights:
            loss = loss * weights

        reduction_fn = Loss.REDUCTION_FN[reduction]
        return reduction_fn(loss)
