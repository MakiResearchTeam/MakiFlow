from __future__ import absolute_import
from abc import abstractmethod
from makiflow.base.maki_entities import MakiTensor


class PathGenerator(object):
    image = 'image'
    mask = 'mask'
    
    @abstractmethod
    def next_element(self) -> dict:
        pass


class MapMethod:
    image = 'image'
    mask = 'mask'
    num_positives = 'num_positives'

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass


class PostMapMethod(MapMethod):
    def __init__(self):
        self._parent_method = None

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass

    def __call__(self, parent_method: MapMethod):
        self._parent_method = parent_method


class SegmentIterator:
    image = 'image'
    mask = 'mask'
    num_positives = 'num_positives'


class GenLayer(MakiTensor):
    def __init__(self, name, input_image):
        self._name = name
        self.image = input_image
        # noinspection PyTypeChecker
        super().__init__(
            data_tensor=self.image,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={}
        )

    def get_shape(self):
        return self.image.get_shape().to_list()

    def get_name(self):
        return self._name

    # noinspection PyMethodMayBeStatic
    def get_params(self):
        return []

    # noinspection PyMethodMayBeStatic
    def get_params_dict(self):
        return {}
