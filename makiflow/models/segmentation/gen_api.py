from __future__ import absolute_import
from makiflow.base import MakiTensor
from abc import abstractmethod
import tensorflow as tf


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
    def load_data(self, paths) -> dict:
        pass

