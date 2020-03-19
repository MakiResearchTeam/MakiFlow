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


class AugmentOp(Augmentor, ABC):
    def __call__(self, data: Augmentor)->Augmentor:
        self._data = data
        self._img_shape = data._get_shape()
        return self

