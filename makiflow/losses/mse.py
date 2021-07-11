import tensorflow as tf

from makiflow.core import Loss
from .single_tensor_loss import SingleTensorLoss


class MSE(SingleTensorLoss):
    def __init__(self, tensor_names: list, label_tensors: dict, reduction=Loss.REDUCTION_MEAN):
        """
        Builds mean squared error loss.

        Parameters
        ----------
        tensor_names : list
            Contains a single tensor name off of which the loss will be built.
        label_tensors : dict
            Dictionary of tensors that supply label data.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        """
        loss_fn = lambda t, lt: MSE.mean_squared_error(t, lt, reduction)
        super().__init__(tensor_names, label_tensors, loss_fn)

    @staticmethod
    def mean_squared_error(tensors, label_tensors, reduction):
        preds = tensors[0]
        labels = label_tensors.get(Loss.LABELS)
        weights = label_tensors.get(Loss.WEIGHTS)

        loss = tf.square(labels - preds)

        if weights:
            loss = loss * weights

        reduction_fn = Loss.REDUCTION_FN[reduction]
        return reduction_fn(loss)
