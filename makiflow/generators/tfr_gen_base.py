from __future__ import absolute_import
from abc import abstractmethod


class TFRMapMethod:
    @abstractmethod
    def read_record(self, serialized_example) -> dict:
        pass


class TFRPostMapMethod(TFRMapMethod):
    def __init__(self):
        self._parent_method = None

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass

    def __call__(self, parent_method: TFRMapMethod):
        self._parent_method = parent_method
        return self


class TFRPathGenerator(object):
    TFRECORD = 'tfrecord'

    @abstractmethod
    def next_element(self) -> dict:
        pass
