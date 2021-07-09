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

import numpy as np
from ..core import BaseInitializer


class RandomNormal(BaseInitializer):

    def __init__(self, mean=0.0, variance=0.01):
        self._mean = mean
        self._variance = variance

    def _create_matrix(self, shape: list, dtype=np.float32):
        w = np.random.normal(loc=self._mean, scale=self._variance, size=shape)
        return w.astype(dtype=dtype, copy=False)


