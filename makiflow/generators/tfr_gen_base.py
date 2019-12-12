from __future__ import absolute_import
from abc import abstractmethod


class TFRMapMethod:
    @abstractmethod
    def read_record(self, serialized_example):
        pass


class TFRPathGenerator(object):
    TFRECORD = 'tfrecord'

    @abstractmethod
    def next_element(self) -> dict:
        pass
