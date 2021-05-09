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

from __future__ import absolute_import
from abc import abstractmethod, ABC
from makiflow.core import InputMakiLayer, MakiRestorable, MakiTensor


class PathGenerator(ABC):
    @abstractmethod
    def next_element(self) -> dict:
        pass


class GenLayer(InputMakiLayer):
    def __init__(self, name, input_tensor):
        self._name = name
        self._input_tensor = input_tensor
        # noinspection PyTypeChecker
        super().__init__(
            data_tensor=input_tensor,
            name=name
        )

    @abstractmethod
    def get_iterator(self):
        pass

    def shape(self):
        return self._input_tensor.shape().as_list()

    def name(self):
        return self._name

    # noinspection PyMethodMayBeStatic
    def get_params(self):
        return []

    # noinspection PyMethodMayBeStatic
    def get_params_dict(self):
        return {}

    def to_dict(self):
        return {
            MakiRestorable.NAME: self.name(),
            MakiTensor.PARENT_TENSOR_NAMES: self.parent_tensor_names(),
            MakiTensor.PARENT_LAYER_INFO: {
                MakiRestorable.TYPE: InputMakiLayer.TYPE,
                MakiRestorable.PARAMS: {
                    MakiRestorable.NAME: self.name(),
                    InputMakiLayer.INPUT_SHAPE: self.shape()
                }
            }
        }
