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

from __future__ import absolute_import
import tensorflow as tf

from makiflow.layers.activation_converter import ActivationConverter
from makiflow.base import MakiLayer, MakiTensor, MakiRestorable, InputMakiLayer
from makiflow.layers.sf_layer import SimpleForwardLayer
import numpy as np


class InputLayer(InputMakiLayer):
    _EXCEPTION_IS_NOT_IMPLEMENTED = 'This functionality is not implemented in the InputLayer.'

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
        self._name = str(name)
        self._input_shape = input_shape

        self._input = tf.placeholder(tf.float32, shape=input_shape, name=self._name)

        super().__init__(
            data_tensor=self._input,
            name=name
        )

    def __call__(self, x):
        raise RuntimeError(InputLayer._EXCEPTION_IS_NOT_IMPLEMENTED)

    def _training_forward(self, x):
        raise RuntimeError(InputLayer._EXCEPTION_IS_NOT_IMPLEMENTED)

    @staticmethod
    def build(params: dict):
        input_shape = params[InputLayer.INPUT_SHAPE]
        name = params[MakiRestorable.NAME]

        return InputLayer(
            name=name,
            input_shape=input_shape
        )

    def to_dict(self):
        return {
            MakiRestorable.NAME: self.get_name(),
            MakiTensor.PARENT_TENSOR_NAMES: self.get_parent_tensor_names(),
            MakiRestorable.FIELD_TYPE: InputMakiLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                InputMakiLayer.INPUT_SHAPE: self.get_shape()
            }
        }


class ReshapeLayer(SimpleForwardLayer):
    TYPE = 'ReshapeLayer'
    NEW_SHAPE = 'new_shape'
    IGNORE_BATCH = 'ignore_batch'

    def __init__(self, new_shape: list, name, ignore_batch=False):
        """
        ReshapeLayer is used to changes size from some input_shape to new_shape (include batch_size and color dimension).

        Parameters
        ----------
        new_shape : list
            Shape of output object.
        name : str
            Name of this layer.
        ignore_batch : bool
            If set to True, the first dimension in the `new_shape` will replace the batch dimension.
            Examples:
            - True - [batch_size, old_shape] -> [new_shape]
            - False - [batch_size, old_shape] -> [batch_size, new_shape]
        """
        self.new_shape = new_shape
        self.ignore_batch = ignore_batch
        super().__init__(
            name, params=[],
            regularize_params=[],
            named_params_dict={}
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                if self.ignore_batch:
                    return tf.reshape(tensor=X, shape=self.new_shape, name=self._name)
                else:
                    bs = X.get_shape().as_list()[0]
                    return tf.reshape(tensor=X, shape=[bs, *self.new_shape], name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        new_shape = params[ReshapeLayer.NEW_SHAPE]
        ignore_batch = params.get(ReshapeLayer.IGNORE_BATCH, False)

        return ReshapeLayer(
            new_shape=new_shape,
            name=name,
            ignore_batch=ignore_batch
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ReshapeLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ReshapeLayer.NEW_SHAPE: self.new_shape,
                ReshapeLayer.IGNORE_BATCH: self.ignore_batch
            }
        }


class MulByAlphaLayer(SimpleForwardLayer):
    TYPE = 'MulByAlphaLayer'
    ALPHA = 'alpha'

    def __init__(self, alpha, name):
        """
        MulByAlphaLayer is used to multiply input MakiTensor on alpha.

        Parameters
        ----------
        alpha : float
            The constant to multiply by.
        name : str
            Name of this layer.
        """
        self.alpha = np.float32(alpha)
        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.math.multiply(X, self.alpha, name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        alpha = float(params[MulByAlphaLayer.ALPHA])

        return MulByAlphaLayer(
            alpha=alpha,
            name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: MulByAlphaLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                MulByAlphaLayer.ALPHA: float(self.alpha),
            }
        }


class SumLayer(MakiLayer):
    TYPE = 'SumLayer'

    def __init__(self, name):
        """
        SumLayer is used sum inputs MakiTensors and give one output MakiTensor.

        Parameters
        ----------
        name : str
            Name of this layer.
        """

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

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

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                # Compare with tf.reduce_sum and tf.add_n, sum(X) works faster in running session
                return sum(X)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        return SumLayer(name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: SumLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
            }
        }


class ConcatLayer(MakiLayer):
    TYPE = 'ConcatLayer'
    AXIS = 'axis'

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
        self.axis = axis

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

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

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.concat(values=X, axis=self.axis, name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        axis = params[ConcatLayer.AXIS]

        return ConcatLayer(
            name=name,
            axis=axis
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ConcatLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ConcatLayer.AXIS: self.axis,
            }
        }


class ZeroPaddingLayer(SimpleForwardLayer):
    TYPE = 'ZeroPaddingLayer'
    PADDING = 'padding'
    CONSTANT = "CONSTANT"

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

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.pad(
                    tensor=X,
                    paddings=self.padding,
                    mode=ZeroPaddingLayer.CONSTANT,
                    name=self._name
                )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        padding = params[ZeroPaddingLayer.PADDING]

        return ZeroPaddingLayer(
            padding=padding,
            name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ZeroPaddingLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ZeroPaddingLayer.PADDING: self.input_padding,
            }
        }


class GlobalMaxPoolLayer(SimpleForwardLayer):
    TYPE = 'GlobalMaxPoolLayer'
    _ASSERT_WRONG_INPUT_SHAPE = 'Input MakiTensor must have 4 dimensional shape'

    def __init__(self, name):
        """
        Performs global maxpooling.
        NOTICE! After this operation tensor will be flatten.

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                assert (len(X.shape) == 4), GlobalMaxPoolLayer._ASSERT_WRONG_INPUT_SHAPE
                return tf.reduce_max(X, axis=[1, 2], name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        return GlobalMaxPoolLayer(name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: GlobalMaxPoolLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
            }
        }


class GlobalAvgPoolLayer(SimpleForwardLayer):
    TYPE = 'GlobalAvgPoolLayer'
    _ASSERT_WRONG_INPUT_SHAPE = 'Input MakiTensor must have 4 dimensional shape'

    def __init__(self, name):
        """
        Performs global avgpooling.
        NOTICE! After this operation tensor will be flatten.

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                assert (len(X.shape) == 4), GlobalAvgPoolLayer._ASSERT_WRONG_INPUT_SHAPE
                return tf.reduce_mean(X, axis=[1, 2], name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        return GlobalAvgPoolLayer(name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: GlobalAvgPoolLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
            }
        }


class MaxPoolLayer(SimpleForwardLayer):
    TYPE = 'MaxPoolLayer'
    KSIZE = 'ksize'
    STRIDES = 'strides'
    PADDING = 'padding'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

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
            Padding mode for convolution operation.
            Options: MaxPoolLayer.PADDING_SAME which is 'SAME' string
            or MaxPoolLayer.PADDING_VALID 'VALID' (case sensitive).
        name : str
            Name of this layer.
        """
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.nn.max_pool(
                    X,
                    ksize=self.ksize,
                    strides=self.strides,
                    padding=self.padding
                )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        ksize = params[MaxPoolLayer.KSIZE]
        strides = params[MaxPoolLayer.STRIDES]
        padding = params[MaxPoolLayer.PADDING]

        return MaxPoolLayer(
            name=name,
            ksize=ksize,
            strides=strides,
            padding=padding
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: MaxPoolLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                MaxPoolLayer.KSIZE: self.ksize,
                MaxPoolLayer.STRIDES: self.strides,
                MaxPoolLayer.PADDING: self.padding
            }
        }


class AvgPoolLayer(SimpleForwardLayer):
    TYPE = 'AvgPoolLayer'
    KSIZE = 'ksize'
    STRIDES = 'strides'
    PADDING = 'padding'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

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
            Padding mode for convolution operation.
            Options: AvgPoolLayer.PADDING_SAME which is 'SAME' string
            or AvgPoolLayer.PADDING_VALID 'VALID' (case sensitive).
        name : str
            Name of this layer.
        """
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.nn.avg_pool(
                    X,
                    ksize=self.ksize,
                    strides=self.strides,
                    padding=self.padding
                )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        ksize = params[AvgPoolLayer.KSIZE]
        strides = params[AvgPoolLayer.STRIDES]
        padding = params[AvgPoolLayer.PADDING]
        name = params[MakiRestorable.NAME]

        return AvgPoolLayer(
            ksize=ksize,
            strides=strides,
            padding=padding,
            name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: AvgPoolLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                AvgPoolLayer.KSIZE: self.ksize,
                AvgPoolLayer.STRIDES: self.strides,
                AvgPoolLayer.PADDING: self.padding
            }
        }


class UpSamplingLayer(SimpleForwardLayer):
    TYPE = 'UpSamplingLayer'
    SIZE = 'size'

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
        self.size = size

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                t_shape = X.get_shape()
                im_size = (t_shape[1] * self.size[0], t_shape[2] * self.size[1])
                return tf.image.resize_nearest_neighbor(
                    X,
                    im_size
                )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        size = params[UpSamplingLayer.SIZE]

        return UpSamplingLayer(
            name=name,
            size=size
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: UpSamplingLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                UpSamplingLayer.SIZE: self.size
            }
        }


class ActivationLayer(SimpleForwardLayer):
    TYPE = 'ActivationLayer'
    ACTIVATION = 'activation'

    _EXCEPTION_ACTIVATION_INPUT_NONE = "Activation can't None"

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
        if activation is None:
            raise Exception(ActivationLayer._EXCEPTION_ACTIVATION_INPUT_NONE)
        self.f = activation

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return self.f(X, name=self._name)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        activation = ActivationConverter.str_to_activation(params[ActivationLayer.ACTIVATION])
        name = params[MakiRestorable.NAME]

        return ActivationLayer(
            activation=activation,
            name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ActivationLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ActivationLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f)
            }
        }


class FlattenLayer(SimpleForwardLayer):
    TYPE = 'FlattenLayer'

    def __init__(self, name):
        """
        Flattens the input.
        Example: if input is [B1, H1, W1, C1], after this operation it would be [B1, C2], where C2 = H1 * W1 * C1

        Parameters
        ----------
        name : str
            Name of this layer.
        """
        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.contrib.layers.flatten(X)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        return FlattenLayer(name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: FlattenLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name()
            }
        }


class DropoutLayer(SimpleForwardLayer):
    TYPE = 'DropoutLayer'
    P_KEEP = 'p_keep'
    NOISE_SHAPE = 'noise_shape'
    SEED = 'seed'

    def __init__(self, name, p_keep=0.9, noise_shape=None, seed=None):
        """
        Applies Dropout to the input MakiTensor.

        Parameters
        ----------
        p_keep : float
            A deprecated alias for (1-rate).
        seed : int
            A Python integer. Used to create random seeds.
        noise_shape : list
            1D list of int representing the shape of the binary dropout mask that will be multiplied with the input MakiTensor.
            For example, if shape(x) = [k, l, m, n] (BHWC) and noise_shape = [k, 1, 1, n], each batch and channel component will be kept
            independently and each row and column will be kept or not kept together.
        name : str
            Name of this layer.
        """
        self._p_keep = p_keep
        self.noise_shape = noise_shape
        self.seed = seed

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        return X

    def _training_forward(self, X):
        with tf.name_scope(MakiRestorable.TRAINING_MODE):
            with tf.name_scope(self.get_name()):
                return tf.nn.dropout(X, self._p_keep,
                                     noise_shape=self.noise_shape,
                                     seed=self.seed,
                                     )

    @staticmethod
    def build(params: dict):
        p_keep = params[DropoutLayer.P_KEEP]
        name = params[MakiRestorable.NAME]
        noise_shape = params[DropoutLayer.NOISE_SHAPE]
        seed = params[DropoutLayer.SEED]

        return DropoutLayer(
            p_keep=p_keep,
            name=name,
            noise_shape=noise_shape,
            seed=seed
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: DropoutLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                DropoutLayer.P_KEEP: self._p_keep,
                DropoutLayer.NOISE_SHAPE: self.noise_shape,
                DropoutLayer.SEED: self.seed
            }
        }


class ResizeLayer(SimpleForwardLayer):
    TYPE = 'ResizeLayer'

    INTERPOLATION_BILINEAR = 'bilinear'
    INTERPOLATION_NEAREST_NEIGHBOR = 'nearest_neighbor'
    INTERPOLATION_AREA = 'area'
    INTERPOLATION_BICUBIC = 'bicubic'

    FIELD_INTERPOLATION = 'interpolation'
    NEW_SHAPE = 'new_shape'
    ALIGN_CORNERS = 'align_corners'

    _EXCEPTION_INTERPOLATION_IS_NOT_FOUND = "Interpolation {} don't exist"

    def __init__(self, new_shape: list, name, interpolation='bilinear', align_corners=False):
        """
        ResizeLayer resize input MakiTensor to new_shape shape.
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
        self.align_corners = align_corners
        self.interpolation = interpolation

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                if self.interpolation == ResizeLayer.INTERPOLATION_BILINEAR:
                    return tf.image.resize_bilinear(
                        X,
                        self.new_shape,
                        align_corners=self.align_corners,
                        name=self._name,
                    )
                elif self.interpolation == ResizeLayer.INTERPOLATION_NEAREST_NEIGHBOR:
                    return tf.image.resize_nearest_neighbor(
                        X,
                        self.new_shape,
                        align_corners=self.align_corners,
                        name=self._name,
                    )
                elif self.interpolation == ResizeLayer.INTERPOLATION_AREA:
                    return tf.image.resize_area(
                        X,
                        self.new_shape,
                        align_corners=self.align_corners,
                        name=self._name,
                    )
                elif self.interpolation == ResizeLayer.INTERPOLATION_BICUBIC:
                    return tf.image.resize_bicubic(
                        X,
                        self.new_shape,
                        align_corners=self.align_corners,
                        name=self._name,
                    )
                else:
                    raise Exception(
                        ResizeLayer._EXCEPTION_INTERPOLATION_IS_NOT_FOUND.format(self.interpolation)
                    )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        new_shape = params[ResizeLayer.NEW_SHAPE]
        name = params[MakiRestorable.NAME]
        align_corners = params[ResizeLayer.ALIGN_CORNERS]
        interpolation = params[ResizeLayer.FIELD_INTERPOLATION]

        return ResizeLayer(
            interpolation=interpolation,
            new_shape=new_shape,
            name=name,
            align_corners=align_corners
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ResizeLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ResizeLayer.FIELD_INTERPOLATION: self.interpolation,
                ResizeLayer.NEW_SHAPE: self.new_shape,
                ResizeLayer.ALIGN_CORNERS: self.align_corners,
            }
        }


class L2NormalizationLayer(SimpleForwardLayer):
    TYPE = 'L2NormalizationLayer'
    EPS = 'eps'

    def __init__(self, name, eps=1e-12):
        """
        This layer was introduced in 'PARSENET: LOOKING WIDER TO SEE BETTER'.
        Performs L2 normalization along feature dimension.
        """
        self._eps = eps

        super().__init__(name, params=[],
                         regularize_params=[],
                         named_params_dict={}
                         )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                return tf.math.l2_normalize(
                    x=X, epsilon=self._eps, axis=-1, name=self._name
                )

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        eps = params[L2NormalizationLayer.EPS]

        return L2NormalizationLayer(
            name=name,
            eps=eps
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: L2NormalizationLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                L2NormalizationLayer.EPS: self._eps
            }
        }


class UnTrainableLayerAddress:
    ADDRESS_TO_CLASSES = {
        InputLayer.TYPE: InputLayer,
        ReshapeLayer.TYPE: ReshapeLayer,
        MulByAlphaLayer.TYPE: MulByAlphaLayer,
        SumLayer.TYPE: SumLayer,

        ConcatLayer.TYPE: ConcatLayer,
        ZeroPaddingLayer.TYPE: ZeroPaddingLayer,
        GlobalMaxPoolLayer.TYPE: GlobalMaxPoolLayer,
        GlobalAvgPoolLayer.TYPE: GlobalAvgPoolLayer,

        MaxPoolLayer.TYPE: MaxPoolLayer,
        AvgPoolLayer.TYPE: AvgPoolLayer,

        UpSamplingLayer.TYPE: UpSamplingLayer,
        ActivationLayer.TYPE: ActivationLayer,

        FlattenLayer.TYPE: FlattenLayer,
        DropoutLayer.TYPE: DropoutLayer,
        ResizeLayer.TYPE: ResizeLayer,
        L2NormalizationLayer.TYPE: L2NormalizationLayer,
    }


from makiflow.base.maki_entities.maki_builder import MakiBuilder

MakiBuilder.register_layers(UnTrainableLayerAddress.ADDRESS_TO_CLASSES)

del MakiBuilder
