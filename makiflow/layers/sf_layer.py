from __future__ import absolute_import
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from copy import copy

from makiflow.save_recover.activation_converter import ActivationConverter
from makiflow.base import MakiLayer, MakiTensor

class SimpleForwardLayer(MakiLayer):
    def __call__(self, x):
        data = x.get_data_tensor()
        data = self._forward(data)

        parent_tensor_names = [x.get_name()]
        previous_tensors = copy(x.get_previous_tensors())
        previous_tensors.update(x.get_self_pair())
        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names=parent_tensor_names,
            previous_tensors=previous_tensors,
        )
        return maki_tensor

    @abstractmethod
    def _forward(self, X):
        pass