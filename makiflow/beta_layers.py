from __future__ import absolute_import
from makiflow.base import MakiLayer
from makiflow.base import MakiTensor
import tensorflow as tf
import numpy as np
from copy import copy
# Merge dict
from collections import ChainMap

from makiflow.save_recover.activation_converter import ActivationConverter


class SumMakiLayer(MakiLayer):
    def __init__(self,name='sum'):
        MakiLayer.__init__(self)
        self.name = name
    
    def forward(self,X,is_training):
        return sum(X)
    
    def __call__(self,x:list) -> MakiTensor :
        data = [i.get_data_tensor() for i in x]
        data = self.forward(data,is_training=True)
        parent_tensor_names = [i.get_name() for i in x]
        arr = [i.get_previous_tensors() for i in x] + [i.get_self_pair() for i in x]
        previous_tensors = dict(ChainMap(*arr))
        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names = parent_tensor_names,
            previous_tensors=previous_tensors,
        )
        return maki_tensor


class InputLayer(MakiTensor):
    def __init__(self, input_shape, name='Input'):
        self.params = []
        self._name = str(name)
        self.__input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape, name=self._name)
        super().__init__(
            data_tensor=self.input,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={},
        )

    def get_shape(self):
        return self.__input_shape
    
    def get_name(self):
        return self._name

    def get_params(self):
        return []

    def get_params_dict(self):
        return {}

    def to_dict(self):
        return {
            'type': 'InputLayer',
            'params': {
                'name': self._name,
                'input_shape': self.__input_shape
            }
        }


class DenseMakiLayer(MakiLayer):
    def __call__(self, x: MakiTensor) -> MakiTensor:
        data = x.get_data_tensor()

        data = tf.matmul(data, self.W) + self.b
        if self.f is not None:
            data = self.f(data)

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

    def __init__(self, input_shape, output_shape, name, activation=tf.nn.relu, init_type='xavier',
                 W=None, b=None):
        """
        :param input_shape - number represents input shape. Example: 500.
        :param output_shape - number represents output shape. You can treat it as number of neurons. Example: 100.
        :param activation - activation function. Set None if you don't need activation.
        :param init_type - name of the weights initialization way: `xavier` or `lasange`. For relu like activations
            `xavier` initialization performs better.
        :param W - matrix weights. Used for initialisation dense weights with pretrained weights.
        :param b - bias weights. Used for initialisation dense bias with pretrained bias.
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.f = activation

        if W is None:
            W = np.random.randn(input_shape, output_shape)
            # Perform Xavier initialization
            if init_type == 'xavier':
                W /= (input_shape + output_shape) / 2
            # Perform Lasange initialization
            else:
                W *= np.sqrt(12 / (input_shape + output_shape))

        if b is None:
            b = np.zeros(output_shape)

        name = str(name)
        self.name_dense = 'DenseMat{}x{}_id_'.format(input_shape, output_shape) + name
        self.name_bias = 'DenseBias{}x{}_id_'.format(input_shape, output_shape) + name

        self.W = tf.Variable(W.astype(np.float32), name=self.name_dense)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        params = [self.W, self.b]
        named_params_dict = {self.name_dense: self.W, self.name_bias: self.b}
        super().__init__(name, params, named_params_dict)

    def _training_forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.f is None:
            return out
        return self.f(out)

    def to_dict(self):
        return {
            'type': 'DenseLayer',
            'params': {
                'name': self._name,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'activation': ActivationConverter.activation_to_str(self.f)
            }
        }
