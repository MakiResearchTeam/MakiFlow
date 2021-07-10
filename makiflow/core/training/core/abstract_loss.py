import abc
from collections import OrderedDict

from makiflow.core.training.core.tensor_provider import TensorProvider


class AbstractLoss(object):
    _id = 1

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        setattr(obj, f'_{__class__.__name__}__id', AbstractLoss._id)
        AbstractLoss._id += 1
        return obj

    def __init__(self, label_tensors):
        # We need to update tensor names to avoid intersection with tensor names in other losses.
        # Have similar names may cause tensor removal which in turn will cause errors.
        upd_label_tensors = {}
        for k, v in label_tensors.items():
            upd_label_tensors[self.get_tensor_name(k)] = v
        self.__unique_label_tensors = OrderedDict(upd_label_tensors)
        # The loss construction method will expect label_tensors to have their names as they were.
        self.__label_tensors = label_tensors
        # The loss tensor will be saved in this var
        self._loss = None

    @abc.abstractmethod
    def build(self, tensor_provider: TensorProvider):
        pass

    @property
    def label_tensors(self) -> OrderedDict:
        return self.__label_tensors.copy()

    @property
    def unique_label_tensors(self) -> OrderedDict:
        return self.__unique_label_tensors.copy()

    @property
    def loss(self):
        return self._loss

    # noinspection PyUnresolvedReferences
    def get_tensor_name(self, tensor_name):
        return tensor_name + '_' + self.__class__.__name__ + '_' + str(self.__id)


del TensorProvider
