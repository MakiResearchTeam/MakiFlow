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
from ..core import BaseGaussianInitializer


class HeGrad(BaseGaussianInitializer):

    def _create_matrix(self, shape: list, dtype=np.float32):
        w = super()._create_matrix(shape=shape, dtype=dtype)
        w *= np.sqrt(2.0 / (np.prod(shape[:-2]) * shape[-1]))
        return w


