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


class InputMakiLayer(MakiTensor, ABC):
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
        self._name = name
        self._params = []
        self._named_params_dict = {}
        self._regularize_params = []
        super().__init__(
            data_tensor=data_tensor,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={},
        )

    def get_params(self):
        """
        :return
        ----------
        Trainable parameters of the layer.
        """
        return self._params

    def get_params_dict(self):
        """
        This code imitates MakiLayer API for the compatibility sake.
        This data is used for correct saving and loading models using TensorFlow checkpoint files.
        """
        return self._named_params_dict

    def get_params_regularize(self):
        """
        This code imitates MakiLayer API for the compatibility sake.
        This data is used for collect params for regularisation.
        :return:
        list
            List of parameters for regularisation.
        """
        return self._regularize_params

    def get_name(self):
        return self._name