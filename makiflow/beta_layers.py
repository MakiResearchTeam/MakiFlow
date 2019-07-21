from abc import abstractmethod
import tensorflow as tf
import numpy as np
from copy import copy

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
    def __init__(self, data_tensor: tf.Tensor, binded_layer, last_tensor_name: str, previous_tensors: dict):
        self.__data_tensor: tf.Tensor = data_tensor
        self.__name: str = binded_layer.name
        self.__last_tensor_name: str = last_tensor_name
        self.__binded_layer = binded_layer
        self.__previous_tensors: dict = previous_tensors

    def get_data_tensor(self):
        return self.__data_tensor

    def get_binded_layer(self):
        return self.__binded_layer

    def get_last_tensor_name(self):
        return self.__last_tensor_name

    def get_previous_tensors(self) -> dict:
        return self.__previous_tensors

    def get_self_pair(self) -> dict:
        return {self.__name: self}


class MakiOperation:
    @abstractmethod
    def __call__(self, x: MakiTensor)-> MakiTensor:
        pass


class InputLayer(MakiTensor):

    def __init__(self, input_shape, name='Input'):
        self.name = str(name)
        self.__input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape, name=self.name)
        super().__init__(
            data_tensor=self.input,
            name=self.name,
            binded_layer=self,
            previous_tensors={}
        )

    def get_shape(self):
        return self.__input_shape


class DenseLayer(Layer, MakiOperation):
    def __call__(self, x: MakiTensor) -> MakiTensor:
        data = x.get_data_tensor()
        data = self.forward(data)
        previous_tensors = copy(x.get_previous_tensors())
        previous_tensors.update(x.get_self_pair())
        maki_tensor = MakiTensor(
            data_tensor=data,
            name=self.name,
            binded_layer=self,
            previous_tensors=previous_tensors
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

