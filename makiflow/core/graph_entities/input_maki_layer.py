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

from makiflow.core.graph_entities.maki_tensor import MakiTensor
from abc import ABC


class InputMakiLayer(MakiTensor, ABC):
    TYPE = 'InputLayer'
    INPUT_SHAPE = 'input_shape'

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
            parent_tensor_names=[],
            previous_tensors={},
        )

    def get_params(self):
        """
        Return
        ----------
        list
            Trainable parameters of this layer.
        """
        return self._params

    def get_params_dict(self):
        """
        This code imitates MakiLayer API for the compatibility sake.
        This data is used for correct saving and loading models using TensorFlow checkpoint files.

        Return
        ----------
        dict
            Dictionary that store name of tensor and tensor itself of this layer.
        """
        return self._named_params_dict

    def get_params_regularize(self):
        """
        This data is used for collect params for regularisation.
        Some of the parameters, like bias, are preferred not to be regularized since it can cause underfitting.
        Thus, it makes sense to track parameters that are being regularized and that are not.

        Return
        ----------
        list
            List of parameters to be regularized.
        """
        return self._regularize_params

    def get_name(self):
        return self._name
