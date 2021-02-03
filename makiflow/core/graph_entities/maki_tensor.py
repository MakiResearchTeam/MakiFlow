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

    OBJ2STR = "MakiTensor(name={}, shape={}, dtype={})"
    OBJ2REPR = "<mf.trainer.MakiTensor 'name={}' shape={} dtype={}>"

    def __init__(self, data_tensor: tf.Tensor, parent_layer, parent_tensor_names: list,
                 previous_tensors: dict, name=None, index=None):
        """
        Parameters
        ----------
        data_tensor : tf.Tensor
            Actual data tensor.
        parent_layer : MakiLayer
            Layer that produced this MakiTensor.
        parent_tensor_names : list
            Name of the MakiTensors used to produce this MakiTensor. I.e., names of the
            MakiTensors that were inputs to the `parent_layer`.
        previous_tensors : dict
            Dictionary of all the MakiTensors that appeared in the graph before creation of this MakiTensor/
        name : str
            Custom name for this MakiTensor
        index : int
            Layer can produce a list of MakiTensors. This is the index of this MakiTensor from such a list.
            If the index is None, the `parent_layer` never produces a list of MakiTensors,
            hence there is no index value.
        """
        self._data_tensor: tf.Tensor = data_tensor
        if name is not None:
            self._name = name
        else:
            self._name: str = parent_layer.get_name()
        self._parent_tensor_names = parent_tensor_names
        self._parent_layer = parent_layer
        self._previous_tensors: dict = previous_tensors
        self._index = index

    def get_data_tensor(self):
        return self._data_tensor

    def get_parent_layer(self):
        """
        Returns
        -------
        Layer
            Layer which produced current MakiTensor.
        """
        return self._parent_layer

    def get_parent_tensors(self) -> list:
        """
        Returns
        -------
        list of MakiTensors
            MakiTensors that were used for creating current MakiTensor.
        """
        parent_tensors = []
        for name in self._parent_tensor_names:
            parent_tensors += [self._previous_tensors[name]]
        return parent_tensors

    def get_parent_tensor_names(self):
        return self._parent_tensor_names

    def get_previous_tensors(self) -> dict:
        """
        Returns
        -------
        dict of MakiTensors
            All the MakiTensors that appear earlier in the computational graph.
            The dictionary contains pairs: { name of the tensor: MakiTensor }.
        """
        return self._previous_tensors

    def get_shape(self):
        return self._data_tensor.get_shape().as_list()

    def get_self_pair(self) -> dict:
        return {self._name: self}

    def __str__(self):
        name = self._name
        shape = self.get_shape()
        dtype = self._data_tensor.dtype.name

        return MakiTensor.OBJ2STR.format(name, shape, dtype)

    def __repr__(self):
        name = self._name
        shape = self.get_shape()
        dtype = self._data_tensor.dtype.name

        return MakiTensor.OBJ2REPR.format(name, shape, dtype)

    def get_name(self):
        return self._name

    def get_index(self):
        return self._index

    def to_dict(self):
        parent_layer_dict = self._parent_layer.to_dict()
        return {
            MakiTensor.NAME: self._name,
            MakiTensor.PARENT_TENSOR_NAMES: self._parent_tensor_names,
            MakiTensor.PARENT_LAYER_INFO: parent_layer_dict
        }

    def eval(self, feed_dict: dict, sess):
        tf_feed_dict = {}
        for makitensor, datatensor in feed_dict.items():
            tf_feed_dict[makitensor.get_data_tensor()] = datatensor

        run_tensor = self.get_data_tensor()
        return sess.run(
            run_tensor,
            feed_dict=tf_feed_dict
        )

