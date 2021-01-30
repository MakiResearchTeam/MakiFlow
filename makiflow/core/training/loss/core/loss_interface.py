from abc import abstractmethod
from collections import OrderedDict

from .tensor_provider import TensorProvider


class LossInterface:
    @abstractmethod
    def build(self, tensor_provider: TensorProvider):
        pass

    @abstractmethod
    def get_label_tensors(self) -> OrderedDict:
        pass


del TensorProvider
