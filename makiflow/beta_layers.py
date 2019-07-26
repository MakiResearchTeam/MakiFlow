from abc import abstractmethod
import tensorflow as tf
import numpy as np
from copy import copy
# Merge dict
from collections import ChainMap

class Layer(object):
    def __init__(self):
        self.params = []
        self.named_params_dict = {}

    def forward(self, X, is_training=False):
        pass

    def get_params(self):
        """
        This data is used for initializing.
        """
        return self.params

    def get_params_dict(self):
        """
        This data is used for correct saving and loading models using TensorFlow checkpoint files.
        """
        return self.named_params_dict

    def to_dict(self):
        """
        This data is used for converting the model's architecture to json.json file.
        """
        pass


class MakiTensor:
    def __init__(self, data_tensor: tf.Tensor, parent_layer, parent_tensor_names: list, previous_tensors: dict):
        self.__data_tensor: tf.Tensor = data_tensor
        self.__name: str = parent_layer.name
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

    def get_parent_tensors(self)->list:
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
        """
        return self.__previous_tensors

    def get_self_pair(self) -> dict:
        return {self.__name: self}
    
    def get_name(self):
        return self.__name


class MakiOperation:
    @abstractmethod
    def __call__(self, x: MakiTensor)-> MakiTensor:
        pass

class SumLayer(Layer, MakiOperation):
    def __init__(self,name='sum'):
        Layer.__init__(self)
        self.name = name
    
    def forward(self,X,is_training):
        return sum(X)
    
    def __call__(self,x:list) -> MakiTensor :
        data = [i.get_data_tensor() for i in x]
        data = self.forward(data,is_training=True)
        parent_tensor_names = [i.get_name() for i in x]
        arr = [i.get_previous_tensors() for i in x] + [i.get_self_pair() for i in x]
        previous_tensors = dict(ChainMap(*arr))
        #previous_tensors.update([i.get_self_pair() for i in x])
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
        self.name = str(name)
        self.__input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape, name=self.name)
        super().__init__(
            data_tensor=self.input,
            parent_layer=self,
            parent_tensor_names = None,
            previous_tensors={},
        )

    def get_shape(self):
        return self.__input_shape
    
    def get_params(self):
        return self.params
    
    def get_params_dict(self):
        return []


class DenseLayer(Layer, MakiOperation):
    def __call__(self, x: MakiTensor) -> MakiTensor:
        data = x.get_data_tensor()
        data = self.forward(data)
        parent_tensor_names = [x.get_name()]
        previous_tensors = copy(x.get_previous_tensors())
        previous_tensors.update(x.get_self_pair())
        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names = parent_tensor_names,
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
        Layer.__init__(self)
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

        self.name = str(name)
        self.name_dense = 'DenseMat{}x{}_id_'.format(input_shape, output_shape) + str(name)
        self.name_bias = 'DenseBias{}x{}_id_'.format(input_shape, output_shape) + str(name)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_dense)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        self.params = [self.W, self.b]
        self.named_params_dict = {self.name_dense: self.W, self.name_bias: self.b}

    def forward(self, X, is_training=False):
        out = tf.matmul(X, self.W) + self.b
        if self.f is None:
            return out
        return self.f(out)

