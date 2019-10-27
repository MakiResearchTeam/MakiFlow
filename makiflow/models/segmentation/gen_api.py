from __future__ import absolute_import
from abc import abstractmethod


class SegmentationGenerator(object):
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
