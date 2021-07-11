import tensorflow as tf
import numpy as np

from ..core import TensorProvider, AbstractLoss
from .utils import filter_tensors


class ConstantLoss(AbstractLoss):
    def __init__(self, const):
        super().__init__({})
        self._const = const

    def build(self, tensor_provider: TensorProvider):
        return self._const


class ModularLoss(AbstractLoss):
    def __init__(self, loss1: AbstractLoss, loss2: AbstractLoss, op):
        loss1 = self.check_loss(loss1)
        loss2 = self.check_loss(loss2)
        if loss1.loss is not None or loss2.loss is not None:
            print('One of the losses has already been built. Make sure it wont cause inconsistencies in graph '
                  'computations as the built loss may require data supply from already unreachable tensors.\n'
                  'To avoid possible errors it is better to recreate the loss objects.')
        super(ModularLoss, self).__init__(filter_tensors(loss1.unique_label_tensors, loss2.unique_label_tensors))
        self.parent_losses = loss1, loss2
        self.op = op

    def build(self, tensor_provider):
        if self.loss is not None:
            return self.loss

        loss1 = self.parent_losses[0].build(tensor_provider)
        loss2 = self.parent_losses[1].build(tensor_provider)

        self._loss = self.op(loss1, loss2)
        return self.loss

    def check_loss(self, other) -> AbstractLoss:
        if isinstance(other, int) or isinstance(other, float) \
                or isinstance(other, tf.Tensor) or isinstance(other, np.float32):
            other = ConstantLoss(other)
        else:
            ValueError(f'Unsupported type for ModularLoss: {type(other)}.'
                       f'Supported types are: int, float, tf.Tensor, np.float32')
        return other

    def __add__(self, other):
        return ModularLoss(self, other, lambda x, y: x + y)

    def __mul__(self, other):
        return ModularLoss(self, other, lambda x, y: x * y)

    def __div__(self, other):
        return ModularLoss(self, other, lambda x, y: x / y)

    def __truediv__(self, other):
        return ModularLoss(self, other, lambda x, y: x / y)


class Loss(AbstractLoss):
    LABELS = 'labels'
    WEIGHTS = 'weights'
    REDUCTION_MEAN = 0
    REDUCTION_SUM = 1

    REDUCTION_FN = {
        0: tf.reduce_mean,
        1: tf.reduce_sum
    }

    def __init__(self, tensor_names: list, label_tensors: dict, loss_fn):
        """
        Builds loss using 'loss_fn' by providing it the required tensors from the model
        (according to `tensor_names`) and `label_tensors`.

        Parameters
        ----------
        tensor_names : list
            List of tensors from the model.
        label_tensors : dict
            Dictionary of pairs { 'tensor_name': tf.Tensor }. Usually tf.Tensor is a placeholder or some other
            data source that provides labels for loss computation.
        loss_fn : function
            Function with the following signature: (tensors: list, label_tensors: dict), where tensors are the
            tensors gathered from the model in accordance to `tensor_names` and label_tensors is the `label_tensors`
            dictionary which passed as is.
        """
        super(Loss, self).__init__(label_tensors)
        self.tensor_names = tensor_names
        self.loss_fn = loss_fn

    def build(self, tensor_provider):
        """
        Build the loss and returns tf.Tensor.
        """
        if self.loss is not None:
            return self.loss

        tensors = []
        for t_name in self.tensor_names:
            tensors += [tensor_provider.get_traingraph_tensor(t_name)]

        self._loss = self.loss_fn(tensors, self.label_tensors)
        return self.loss

    def __add__(self, other):
        return ModularLoss(self, other, lambda x, y: x + y)

    def __mul__(self, other):
        return ModularLoss(self, other, lambda x, y: x * y)

    def __div__(self, other):
        return ModularLoss(self, other, lambda x, y: x / y)

    def __truediv__(self, other):
        return ModularLoss(self, other, lambda x, y: x / y)


if __name__ == '__main__':
    loss1 = Loss(None, {})
    loss2 = Loss(None, {})
    print((loss1 + loss2) * loss1 * 2 + 1)
