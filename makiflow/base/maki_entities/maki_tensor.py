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

import tensorflow as tf


class MakiTensor:
    NAME = 'name'
    PARENT_TENSOR_NAMES = 'parent_tensor_names'
    PARENT_LAYER_INFO = 'parent_layer_info'

    def __init__(self, data_tensor: tf.Tensor, parent_layer, parent_tensor_names: list,
                 previous_tensors: dict):
        self.__data_tensor: tf.Tensor = data_tensor
        self.__name: str = parent_layer.get_name()
        self.__parent_tensor_names = parent_tensor_names
        self.__parent_layer = parent_layer
        self.__previous_tensors: dict = previous_tensors

    def get_data_tensor(self):
        return self.__data_tensor

    def get_parent_layer(self):
        """
        Returns
        -------
        Layer
            Layer which produced current MakiTensor.
        """
        return self.__parent_layer

    def get_parent_tensors(self) -> list:
        """
        Returns
        -------
        list of MakiTensors
            MakiTensors that were used for creating current MakiTensor.
        """
        parent_tensors = []
        for name in self.__parent_tensor_names:
            parent_tensors += [self.__previous_tensors[name]]
        return parent_tensors

    def get_parent_tensor_names(self):
        return self.__parent_tensor_names

    def get_previous_tensors(self) -> dict:
        """
        Returns
        -------
        dict of MakiTensors
            All the MakiTensors that appear earlier in the computational graph.
            The dictionary contains pairs: { name of the tensor: MakiTensor }.
        """
        return self.__previous_tensors

    def get_shape(self):
        return self.__data_tensor.get_shape().as_list()

    def get_self_pair(self) -> dict:
        return {self.__name: self}

    def __str__(self):
        name = self.__name
        shape = self.get_shape()
        dtype = self.__data_tensor._dtype.name

        return f"MakiTensor(name={name}, shape={shape}, dtype={dtype})"

    def __repr__(self):
        name = self.__name
        shape = self.get_shape()
        dtype = self.__data_tensor._dtype.name

        return f"<mf.base.MakiTensor 'name={name}' shape={shape} dtype={dtype}>"

    def get_name(self):
        return self.__name

    def to_dict(self):
        parent_layer_dict = self.__parent_layer.to_dict()
        return {
            MakiTensor.NAME: self.__name,
            MakiTensor.PARENT_TENSOR_NAMES: self.__parent_tensor_names,
            MakiTensor.PARENT_LAYER_INFO: parent_layer_dict
        }

