# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

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
    def __init__(self, tfrecords, init_shuffle=False):
        """
        Generator for the tfrecord pipeline which gives next tfrecord in a cyclic order.
        After each cycle data is randomly shuffled.

        Parameters
        ----------
        tfrecords : list
            List of paths to the tfrecords.
        init_shuffle : bool
            If True, shuffle of tfrecords will be performed before the cycle begins
        """
        if init_shuffle:
            self._tfrecords = shuffle(tfrecords)
        else:
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
