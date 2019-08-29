from abc import ABC, abstractmethod


class Augmentor(ABC):
    def __init__(self):
        # Must be set in the data provider
        self._img_shape = None

    @abstractmethod
    def get_data(self):
        pass

    def _get_shape(self):
        return self._img_shape



class AugmentOp(Augmentor):
    @abstractmethod
    def __call__(self, data: Augmentor):
        self._data = data
        self._img_shape = data._get_shape()

