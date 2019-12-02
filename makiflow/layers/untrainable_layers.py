from __future__ import absolute_import
import tensorflow as tf

from makiflow.layers.activation_converter import ActivationConverter
from makiflow.base import MakiLayer, MakiTensor
from makiflow.layers.sf_layer import SimpleForwardLayer


class InputLayer(MakiTensor):
    def __init__(self, input_shape, name):
        """
        InputLayer is used to instantiate MakiFlow tensor.

        Parameters
        ----------
        input_shape : list
            Shape of input object.
        name : str
            Name of this layer.
        """
        self.params = []
        self._name = str(name)
        self._input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape, name=self._name)
        super().__init__(
            data_tensor=self.input,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={},
        )

    def get_shape(self):
        return self._input_shape

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
                'input_shape': self._input_shape
            }
        }


class ReshapeLayer(SimpleForwardLayer):
    def __init__(self, new_shape: list, name):
        """
        ReshapeLayer is used to changes size from some input_shape to new_shape (include batch_size and color dimension).

        Parameters
        ----------
        new_shape : list
            Shape of output object.
        name : str
            Name of this layer.
        """
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


class MulByAlphaLayer(SimpleForwardLayer):
    def __init__(self, alpha, name):
        """
        MulByAlphaLayer is used to multiply input MakiTensor on alpha.

        Parameters
        ----------
        alpha : int
            The constant to multiply by.
        name : str
            Name of this layer.
        """
        self.alpha = tf.constant(alpha)
        super().__init__(name, [], {})

    def _forward(self, X):
        return X * self.alpha

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'MulByAlphaLayer',
            'params': {
                'name': self.get_name(),
                'alpha': self.alpha,
            }
        }


class SumLayer(MakiLayer):
    def __init__(self, name):
        """
        SumLayer is used sum inputs MakiTensors and give one output MakiTensor.

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})

    def __call__(self, x: list):
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

    def _forward(self, X):
        return sum(X)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'SumLayer',
            'params': {
                'name': self._name,
            }
        }


class ConcatLayer(MakiLayer):
    def __init__(self, name, axis=3):
        """
        ConcatLayer is used concatenate input MakiTensors along certain axis.

        Parameters
        ----------
        axis : int
            Dimension along which to concatenate.
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})
        self.axis = axis

    def __call__(self, x: list):
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

    def _forward(self, X):
        return tf.concat(values=X, axis=self.axis)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'ConcatLayer',
            'params': {
                'name': self._name,
                'axis': self.axis,
            }
        }


class ZeroPaddingLayer(SimpleForwardLayer):
    def __init__(self, padding, name):
        """
        ZeroPaddingLayer adds rows and columns of zeros
        at the top, bottom, left and right side of an image tensor.

        Parameters
        ----------
        padding : list
            List the number of additional rows and columns in the appropriate directions. 
            For example like [ [top,bottom], [left,right] ]
        name : str
            Name of this layer.
        """
        assert (len(padding) == 2)
        self.input_padding = padding
        self.padding = [[0, 0], padding[0], padding[1], [0, 0]]
        super().__init__(name, [], {})

    def _forward(self, X):
        return tf.pad(
            tensor=X,
            paddings=self.padding,
            mode="CONSTANT",
        )

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'ZeroPaddingLayer',
            'params': {
                'name': self._name,
                'padding': self.input_padding,
            }
        }


class GlobalMaxPoolLayer(SimpleForwardLayer):
    def __init__(self, name):
        """
        Performs global maxpooling.
        NOTICE! After this operation tensor will be flatten.

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})

    def _forward(self, X):
        assert (len(X.shape) == 4)
        return tf.reduce_max(X, axis=[1, 2])

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
    def __init__(self, name):
        """
        Performs global avgpooling.
        NOTICE! After this operation tensor will be flatten.

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})

    def _forward(self, X):
        assert (len(X.shape) == 4)
        return tf.reduce_mean(X, axis=[1, 2])

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'GlobalAvgPoolLayer',
            'params': {
                'name': self._name,
            }
        }


class MaxPoolLayer(SimpleForwardLayer):
    def __init__(self, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        """
        Max pooling operation for spatial data.
        
        Parameters
        ----------
        ksize : list
            The size of the window for each dimension of the input MakiTensor.
        strides : list
            The stride of the sliding window for each dimension of the input MakiTensor.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive).
        name : str
            Name of this layer.
        """
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
        """
        Average pooling operation for spatial data.
        
        Parameters
        ----------
        ksize : list
            The size of the window for each dimension of the input MakiTensor.
        strides : list
            The stride of the sliding window for each dimension of the input MakiTensor.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive).
        name : str
            Name of this layer.
        """
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
        """
        Upsampling layer which changes height and width of MakiTensor.
        Example: input MakiTensor have shape [N1, H1, W1, C1], after this operation it would be [N1, H2, W2, C1], 
        where H2 = H1 * size[0], W2 = W2 * size[1]
        
        Parameters
        ----------
        size : list
            The upsampling factors for rows and columns.
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})
        self.size = size

    def _forward(self, X):
        t_shape = X.get_shape()
        im_size = (t_shape[1] * self.size[0], t_shape[2] * self.size[1])
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
        """
        Applies an activation function to an input MakiTensor.
        
        Parameters
        ----------
        activation : object
            Activation function from tf.
        name : str
            Name of this layer.
        """
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
        """
        Flattens the input.
        Example: if input is [B1, H1, W1, C1], after this operation it would be [B1, C2], where C2 = H1 * W1 * C1
        
        Parameters
        ----------
        name : str
            Name of this layer.
        """
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
    def __init__(self, name, p_keep=0.9, noise_shape=None, seed=None, rate=None):
        """
        Applies Dropout to the input MakiTensor.
        
        Parameters
        ----------
        p_keep : float
            A deprecated alias for (1-rate).
        rate : float
            A scalar Tensor with the same type as input MakiTensor. The probability that each element of input MakiTensor is discarded.
        seed : int
            A Python integer. Used to create random seeds.
        noise_shape : list
            1D list of int representing the shape of the binary dropout mask that will be multiplied with the input MakiTensor. 
            For example, if shape(x) = [k, l, m, n] (BHWC) and noise_shape = [k, 1, 1, n], each batch and channel component will be kept
            independently and each row and column will be kept or not kept together.
        name : str
            Name of this layer.
        """
        super().__init__(name, [], {})
        self._p_keep = p_keep
        self.noise_shape = noise_shape
        self.seed = seed
        self.rate = rate
        self.forward_p_keep = tf.constant(p_keep, dtype=tf.float32)

    def _forward(self, X):
        return X * self.forward_p_keep

    def _training_forward(self, X):
        return tf.nn.dropout(X, self._p_keep,
                             noise_shape=self.noise_shape,
                             seed=self.seed,
                             rate=self.rate,
                             )

    def to_dict(self):
        return {
            'type': 'DropoutLayer',
            'params': {
                'name': self._name,
                'p_keep': self._p_keep,
                'noise_shape': self.noise_shape,
                'seed': self.seed,
                'rate': self.rate,
            }
        }


class ResizeLayer(SimpleForwardLayer):
    def __init__(self, new_shape: list, name, interpolation='bilinear', align_corners=False):
        """
        ResizeLayer resize input MakiTensor to new_shape shape.
        NOTICE! area interpolation don't have half_pixel_centers parameter
        Parameters
        ----------
        interpolation : str
            One of type resize images. ('bilinear', 'nearest_neighbor', 'area', 'bicubic')
        new_shape : list
            List the number of new shape tensor (Height and Width).
        name : str
            Name of this layer.
        """
        assert (len(new_shape) == 2)
        self.new_shape = new_shape
        self.name = name
        self.align_corners = align_corners
        self.interpolation = interpolation

        super().__init__(name, [], {})

    def _forward(self, X):
        if self.interpolation == 'bilinear':
            return tf.image.resize_bilinear(
                X,
                self.new_shape,
                align_corners=self.align_corners,
                name=self.name,
            )
        elif self.interpolation == 'nearest_neighbor':
            return tf.image.resize_nearest_neighbor(
                X,
                self.new_shape,
                align_corners=self.align_corners,
                name=self.name,
            )
        elif self.interpolation == 'area':
            return tf.image.resize_area(
                X,
                self.new_shape,
                align_corners=self.align_corners,
                name=self.name,
            )
        elif self.interpolation == 'bicubic':
            return tf.image.resize_bicubic(
                X,
                self.new_shape,
                align_corners=self.align_corners,
                name=self.name,
            )
        else:
            raise Exception(f"Interpolation {self.interpolation} don't exist")

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'ResizeLayer',
            'params': {
                'name': self.name,
                'interpolation': self.interpolation,
                'new_shape': self.new_shape,
                'align_corners': self.align_corners,
            }
        }
