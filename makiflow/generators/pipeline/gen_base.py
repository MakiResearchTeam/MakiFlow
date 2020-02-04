from __future__ import absolute_import
from abc import abstractmethod, ABC
from makiflow.base.maki_entities import MakiTensor


class PathGenerator(ABC):
    @abstractmethod
    def next_element(self) -> dict:
        pass


class GenLayer(MakiTensor):
    def __init__(self, name, input_tensor):
        self._name = name
        self._input_tensor = input_tensor
        # noinspection PyTypeChecker
        super().__init__(
            data_tensor=self._input_tensor,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={}
        )

    @abstractmethod
    def get_iterator(self):
        pass

    def get_shape(self):
        return self._input_tensor.get_shape().as_list()

    def get_name(self):
        return self._name

    # noinspection PyMethodMayBeStatic
    def get_params(self):
        return []

    # noinspection PyMethodMayBeStatic
    def get_params_dict(self):
        return {}
