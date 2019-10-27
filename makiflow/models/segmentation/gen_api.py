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

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass

