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


class InputLayer(MakiTensor):
    def __init__(self, input_shape, name):
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
            "name": self._name,
            "parent_tensor_names": [],
            'type': 'InputLayer',
            'params': {
                'name': self._name,
                'input_shape': self.__input_shape
            }
        }


class MultiOnAlphaLayer(SimpleForwardLayer):
    def __init__(self,alpha,name):
        self.alpha = alpha
        super().__init__(name,[],{})
    
    def _forward(self,X):
        return X*self.alpha
    
    def _training_forward(self,X):
        return self._forward(X)
    
    def to_dict(self):
        return {
            'type' : 'MultiOnAlphaLayer',
            'params' : {
                'name' : self.get_name(),
                'alpha' : self.alpha,
            }
        }


class ReshapeLayer(SimpleForwardLayer):
    def __init__(self, new_shape: list, name):
        super().__init__(name, [], {})
        self.new_shape = new_shape

    def _forward(self, X):
        return tf.reshape(tensor=X, shape=self.new_shape, name=self.get_name())

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'ReshapeLayer',
            'params': {
                'name': self.get_name(),
                'new_shape': self.new_shape
            }
        }


class SumLayer(MakiLayer):
    def __init__(self,name):
        super().__init__(name, [], {})
    
    def __call__(self,x : list):
        data = [one_tensor.get_data_tensor() for one_tensor in x]
        data = self._forward(data)

        parent_tensor_names = [one_tensor.get_name() for one_tensor in x]
        previous_tensors = {}
        for one_tensor in x:
            previous_tensors.update(one_tensor.get_previous_tensors())
            previous_tensors.update(one_tensor.get_self_pair())

        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names=parent_tensor_names,
            previous_tensors=previous_tensors,
        )
        return maki_tensor

    def _forward(self,X):
        return sum(X)
    
    def _training_forward(self,X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'SumLayer',
            'params': {
                'name': self._name,
            }
        }


class ConcatLayer(MakiLayer):
    def __init__(self,name,axis=3):
        super().__init__(name, [], {})
        self.axis = axis
    
    def __call__(self,x : list):
        data = [one_tensor.get_data_tensor() for one_tensor in x]
        data = self._forward(data)

        parent_tensor_names = [one_tensor.get_name() for one_tensor in x]
        previous_tensors = {}
        for one_tensor in x:
            previous_tensors.update(one_tensor.get_previous_tensors())
            previous_tensors.update(one_tensor.get_self_pair())

        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names=parent_tensor_names,
            previous_tensors=previous_tensors,
        )
        return maki_tensor
    
    def _forward(self,X):
        return tf.concat(values=X,axis=self.axis)
    
    def _training_forward(self,X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'ConcatLayer',
            'params': {
                'name': self._name,
                'axis' : self.axis,
            }
        }


class ConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation=tf.nn.relu,
                 W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int 
            Number of output feature maps (number of filters).
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensetive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.stride = stride
        self.padding = padding
        self.f = activation

        name = str(name)
        self.name_conv = 'ConvKernel{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
        self.name_bias = 'ConvBias{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name

        if W is None:
            W = np.random.randn(*self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        if b is None:
            b = np.zeros(out_f)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        params = [self.W, self.b]
        named_params_dict = {self.name_conv: self.W, self.name_bias: self.b}
        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'ConvLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'stride': self.stride,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f)
            }
            
        }


class UpConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, out_f, name, size=(2, 2), padding='SAME', activation=tf.nn.relu,
                 W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int 
            Number of output feature maps (number of filters).
        size : tuple
            Tuple of two ints - factors of the size of the output feature map.
            Example: feature map with spatial dimension (n, m) will produce
            output feature map of size (a*n, b*m) after performing up-convolution
            with `size` (a, b).
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensetive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        """
        # Shape is different from normal convolution since it's required by 
        # transposed convolution. Outpute feature maps go before input ones.
        self.shape = (kw, kh, out_f, in_f)
        self.size = size
        self.strides = [1, *size, 1]
        self.padding = padding
        self.f = activation

        name = str(name)
        self.name_conv = 'UpConvKernel_{}x{}_out{}_in{}_id_'.format(kw, kh, out_f, in_f) + name
        self.name_bias = 'UpConvBias_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name

        if W is None:
            W = np.random.randn(*self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        if b is None:
            b = np.zeros(out_f)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        params = [self.W, self.b]
        named_params_dict = {self.name_conv: self.W, self.name_bias: self.b}
        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        out_shape = X.get_shape().as_list()
        out_shape[1] *= self.size[0]
        out_shape[2] *= self.size[1]
        out_shape[3] = self.shape[2] # out_f
        conv_out = tf.nn.conv2d_transpose(
            X, self.W, 
            output_shape=out_shape, strides=self.strides, padding=self.padding
        )
        conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'UpConvLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'size': self.size,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f)
            }

        }


class DenseLayer(SimpleForwardLayer):
    def __init__(self, in_d, out_d, name, activation=tf.nn.relu, init_type='xavier',
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

        self.input_shape = in_d
        self.output_shape = out_d
        self.f = activation

        if W is None:
            W = np.random.randn(in_d, out_d)
            # Perform Xavier initialization
            if init_type == 'xavier':
                W /= (in_d + out_d) / 2
            # Perform Lasange initialization
            else:
                W *= np.sqrt(12 / (in_d + out_d))

        if b is None:
            b = np.zeros(out_d)

        name = str(name)
        self.name_dense = 'DenseMat{}x{}_id_'.format(in_d, out_d) + name
        self.name_bias = 'DenseBias{}x{}_id_'.format(in_d, out_d) + name

        self.W = tf.Variable(W.astype(np.float32), name=self.name_dense)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        params = [self.W, self.b]
        named_params_dict = {self.name_dense: self.W, self.name_bias: self.b}
        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.f is None:
            return out
        return self.f(out)

    def _training_forward(self, X):
        return self._forward(X)

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


class BatchNormLayer(SimpleForwardLayer):
    def __init__(self, D, name, decay=0.9,
                 mean=None, var=None, gamma=None, beta=None):
        """
        :param D - number of tensors to be normalized.
        :param mean - batch mean value. Used for initialization mean with pretrained value.
        :param var - batch variance value. Used for initialization variance with pretrained value.
        :param gamma - batchnorm gamma value. Used for initialization gamma with pretrained value.
        :param beta - batchnorm beta value. Used for initialization beta with pretrained value.
        Batch Noramlization Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        gamma and beta are defined by the NN, e.g. they are trainable.
        """
        self.D = D

        if mean is None:
            mean = np.zeros(D)
        if var is None:
            var = np.ones(D)

        # These variables are needed to change the mean and variance of the batch after
        # the batchnormalization: result*gamma + beta
        # beta - offset
        # gamma - scale
        if beta is None:
            beta = np.zeros(D)
        if gamma is None:
            gamma = np.ones(D)

        name = str(name)
        self.name_mean = 'BatchMean{}_id_'.format(D) + name
        self.name_var = 'BatchVar{}_id_'.format(D) + name
        self.name_gamma = 'BatchGamma{}_id_'.format(D) + name
        self.name_beta = 'BatchBeta{}_id_'.format(D) + name

        self.running_mean = tf.Variable(mean.astype(np.float32), trainable=False, name=self.name_mean)
        self.running_variance = tf.Variable(var.astype(np.float32), trainable=False, name=self.name_var)
        self.gamma = tf.Variable(gamma.astype(np.float32), name=self.name_gamma)
        self.beta = tf.Variable(beta.astype(np.float32), name=self.name_beta)

        self.decay = decay

        params = [self.running_mean, self.running_variance, self.gamma, self.beta]
        named_params_dict = {self.name_mean: self.running_mean, self.name_var: self.running_variance,
                                  self.name_gamma: self.gamma, self.name_beta: self.beta}
        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        return tf.nn.batch_normalization(
            X,
            self.running_mean,
            self.running_variance,
            self.beta,
            self.gamma,
            1e-4
        )

    def _training_forward(self, X):
        """
        :param decay - this argument is responsible for how fast batchnorm layer is trained. Values between 0.9 and 0.999 are
        commonly used.
        """
        # These if statements check if we do batchnorm for convolution or dense
        if len(X.shape) == 4:
            # conv
            axes = [0, 1, 2]
        else:
            # dense
            axes = [0]

        batch_mean, batch_var = tf.nn.moments(X, axes=axes)
        update_running_mean = tf.assign(
            self.running_mean,
            self.running_mean * self.decay + batch_mean * (1 - self.decay)
        )
        update_running_variance = tf.assign(
            self.running_variance,
            self.running_variance * self.decay + batch_var * (1 - self.decay)
        )
        with tf.control_dependencies([update_running_mean, update_running_variance]):
            out = tf.nn.batch_normalization(
                X,
                batch_mean,
                batch_var,
                self.beta,
                self.gamma,
                1e-4
            )

        return out

    def to_dict(self):
        return {
            'type': 'BatchNormLayer',
            'params': {
                'name': self._name,
                'D': self.D,
            }
        }


class ZeroPaddingLayer(SimpleForwardLayer):
    def __init__(self,padding,name):
        """
        This layer can add rows and columns of zeros
        at the top, bottom, left and right side of an image tensor.

        Parameters
        ----------
            padding : list
                List the number of additional rows and columns in the appropriate directions. For example like [ [top,bottom], [left,right] ]
                
        """
        self.padding = padding
        super().__ini__(name,[],{})
    
    def _forward(self, X):
        return tf.pad(
            X,
            padding=self.padding,
            "CONSTANT",
        )

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'ZeroPaddingLayer',
            'params': {
                'name': self._name,
                'padding': self.padding,
            }
        }


class GlobalMaxPoolLayer(SimpleForwardLayer):
    def __init__(self,name):
        super().__ini__(name,[],{})
    
    def _forward(self, X):
        assert(len(X.shape) == 4)
        return tf.reduce_max(X,axis=[1,2])

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'GlobalMaxPoolLayer',
            'params': {
                'name': self._name,
            }
        }


class GlobalAvgPoolLayer(SimpleForwardLayer):
    def __init__(self,name):
        super().__ini__(name,[],{})
    
    def _forward(self,X):
        assert(len(X.shape) == 4)
        return tf.reduce_mean(X,axis=[1,2])
    
    def _training_forward(self,x):
        return self._forward(x)
    
    def to_dict(self):
        return {
            'type' : 'GlobalAvgPoolLayer',
            'params' : {
                'name' : self._name,
            }
        }


class MaxPoolLayer(SimpleForwardLayer):
    def __init__(self, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        super().__init__(name, [], {})
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def _forward(self, X):
        return tf.nn.max_pool(
            X,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding
        )

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'MaxPoolLayer',
            'params': {
                'name': self._name,
                'ksize': self.ksize,
                'strides': self.strides,
                'padding': self.padding
            }
        }


class AvgPoolLayer(SimpleForwardLayer):
    def __init__(self, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        super().__init__(name, [], {})
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def _forward(self, X):
        return tf.nn.avg_pool(
            X,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding
        )

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'AvgPoolLayer',
            'params': {
                'name': self._name,
                'ksize': self.ksize,
                'strides': self.strides,
                'padding': self.padding
            }
        }


class UpSamplingLayer(SimpleForwardLayer):
    def __init__(self, name, size=(2, 2)):
        super().__init__(name, [], {})
        self.size = size

    def _forward(self, X):
        t_shape = X.get_shape().as_list()
        im_size = (t_shape[1]*self.size[0], t_shape[2]*self.size[1])
        return tf.image.resize_nearest_neighbor(
            X,
            im_size
        )

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'UpSamplingLayer',
            'params': {
                'name': self._name,
                'size': self.size
            }
        }


class ActivationLayer(SimpleForwardLayer):
    def __init__(self, name, activation=tf.nn.relu):
        super().__init__(name, [], {})
        if activation is None:
            raise Exception("Activation can't None")
        self.f = activation

    def _forward(self, X):
        return self.f(X)

    def _training_forward(self, X):
        return self.f(X)

    def to_dict(self):
        return {
            'type': 'ActivationLayer',
            'params': {
                'name': self._name,
                'activation': ActivationConverter.activation_to_str(self.f)
            }
        }


class FlattenLayer(SimpleForwardLayer):
    def __init__(self, name):
        super().__init__(name, [], {})

    def _forward(self, X):
        return tf.contrib.layers.flatten(X)

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'FlattenLayer',
            'params': {
                'name': self._name
            }
        }


class DropoutLayer(SimpleForwardLayer):
    def __init__(self, name, p_keep=0.9):
        super().__init__(name, [], {})
        self._p_keep = p_keep

    def _forward(self, X):
        return X

    def _training_forward(self, X):
        return tf.nn.dropout(X, self._p_keep)

    def to_dict(self):
        return {
            'type': 'DropoutLayer',
            'params': {
                'name': self._name,
                'p_keep': self._p_keep
            }
        }
