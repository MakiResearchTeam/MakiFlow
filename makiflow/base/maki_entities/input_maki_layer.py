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

from .maki_tensor import MakiTensor
from .maki_layer import MakiLayer
from abc import ABC


class InputMakiLayer(MakiTensor, MakiLayer, ABC):
    TYPE = 'InputLayer'

    def __init__(self, data_tensor, name):
        """
        InputLayer is used to instantiate MakiFlow tensor.

        Parameters
        ----------
        data_tensor : tf.Tensor
            The data tensor.
        name : str
            Name of this layer.
        """

        self._params = []
        self._name = str(name)
        super().__init__(
            data_tensor=data_tensor,
            parent_layer=self,
            parent_tensor_names=[],
            previous_tensors={},
        )
