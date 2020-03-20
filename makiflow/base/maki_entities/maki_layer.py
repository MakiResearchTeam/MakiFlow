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

from abc import abstractmethod, ABC


class MakiRestorable(ABC):
    TYPE = 'Restorable'
    PARAMS = 'params'
    FIELD_TYPE = 'type'
    NAME = 'name'

    @staticmethod
    def build(params: dict):
        """
        Parameters
        ----------
        params : dict
            Dictionary of specific params to build layers.

        Returns
        -------
        MakiLayer
            Specific built layers
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns
        -------
        dictionary
            Contains all the necessary information for restoring the layer object.
        """
        pass


class MakiLayer(MakiRestorable):
    def __init__(self, name, params, named_params_dict):
        self._name = name
        self._params = params
        self._named_params_dict = named_params_dict

    @abstractmethod
    def __call__(self, x):
        """
        Parameters
        ----------
        x: MakiTensor or list of MakiTensors

        Returns
        -------
        MakiTensor or list of MakiTensors
        """
        pass

    @abstractmethod
    def _training_forward(self, x):
        pass

    def get_params(self):
        return self._params

    def get_params_dict(self):
        """
        This data is used for correct saving and loading models using TensorFlow checkpoint files.
        """
        return self._named_params_dict

    def get_name(self):
        return self._name

