from collections import OrderedDict
from abc import abstractmethod

from .core import LossInterface, TensorProvider


class LossOp(LossInterface):
    def __init__(self, loss1: LossInterface, loss2: LossInterface):
        self._losses = [loss1, loss2]

    def build(self, tensor_provider: TensorProvider):
        loss1 = self._losses[0].build(tensor_provider)
        loss2 = self._losses[1].build(tensor_provider)
        return self._op(loss1, loss2)

    @abstractmethod
    def _op(self, loss1, loss2):
        pass

    def get_label_tensors(self) -> OrderedDict:
        dict1 = self._losses[0].get_label_tensors().copy()
        dict2 = self._losses[1].get_label_tensors().copy()
        dict1.update(dict2)
        return dict1


class MulOp(LossOp):
    def _op(self, loss1, loss2):
        return loss1 * loss2


class AddOp(LossOp):
    def _op(self, loss1, loss2):
        return loss1 + loss2


class DivOp(LossOp):
    def _op(self, loss1, loss2):
        return loss1 / loss2
