from abc import abstractmethod

import numpy as np
from sklearn.utils import shuffle

from makiflow.generators.pipeline.gen_base import PathGenerator


class TFRPathGenerator(PathGenerator):
    TFRECORD = 'tfrecord'

    @abstractmethod
    def next_element(self) -> dict:
        pass


class CycleGenerator(TFRPathGenerator):
    def __init__(self, tfrecords):
        """
        Generator for the tfrecord pipeline which gives next tfrecord in a cyclic order.
        After each cycle data is randomly shuffled.

        Parameters
        ----------
        tfrecords : list
            List of paths to the tfrecords.
        """
        self._tfrecords = tfrecords

    def next_element(self):
        n_records = len(self._tfrecords)
        index = 0
        while True:
            el = {
                TFRPathGenerator.TFRECORD: self._tfrecords[index],
            }
            yield el

            index += 1
            if index == n_records:
                index = 0
                self._tfrecords = shuffle(self._tfrecords)


class RandomGeneratorSegment(TFRPathGenerator):
    def __init__(self, tfrecords: list):
        """
        Generator for the SSD pipeline, which gives next tfrecord in random order.

        Parameters
        ----------
        tfrecords : list
            List of paths to the tfrecords.
        """
        self._tfrecords = tfrecords

    def next_element(self):
        while True:
            index = np.random.randint(low=0, high=len(self._tfrecords))

            el = {
                TFRPathGenerator.TFRECORD: self._tfrecords[index],
            }

            yield el