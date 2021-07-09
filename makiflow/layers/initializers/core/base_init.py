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
from abc import abstractmethod, ABC

from .init_factory import InitFabric


class BaseInitializer(ABC, metaclass=InitFabric):

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        checked_shape = self._check_input_shape(shape=shape)
        return self._create_matrix(shape=checked_shape, dtype=dtype)

    def _check_input_shape(self, shape: list) -> list:
        # Check size and type
        if len(shape) == 0 or len(shape) == 1 or (not isinstance(shape, list) and not isinstance(shape, tuple)):
            raise ValueError(f"Input shape must be list or tuple "
                             f"\nand must have len more than 1, but len {len(shape)} was given.")
        shape = list(shape)
        return shape

    @abstractmethod
    def _create_matrix(self, shape: list, dtype=np.float32):
        pass

