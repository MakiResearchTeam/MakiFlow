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
import numpy as np
import tensorflow as tf

from makiflow.base.maki_entities.maki_layer import MakiRestorable
from makiflow.layers.activation_converter import ActivationConverter
from makiflow.layers.sf_layer import SimpleForwardLayer
from makiflow.base import BatchNormBaseLayer
from makiflow.layers.utils import InitConvKernel, InitDenseMat


class ConvLayer(SimpleForwardLayer):
    TYPE = 'ConvLayer'
    SHAPE = 'shape'
    STRIDE = 'stride'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_BIAS = 'ConvBias_{}x{}_in{}_out{}_id_{}'
    NAME_CONV_W = 'ConvKernel_{}x{}_in{}_out{}_id_{}'

    def __init__(self, kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation=tf.nn.relu,
                 kernel_initializer=InitConvKernel.HE, use_bias=True, regularize_bias=False, W=None, b=None):
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
            Padding mode for convolution operation.
            Options: ConvLayer.PADDING_SAME which is 'SAME' string
            or ConvLayer.PADDING_VALID 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, out_f, in_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_conv = ConvLayer.NAME_CONV_W.format(kw, kh, in_f, out_f, name)
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        regularize_params = [self.W]
        if use_bias:
            self.name_bias = ConvLayer.NAME_BIAS.format(kw, kh, in_f, out_f, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            conv_out = tf.nn.conv2d(
                X, self.W,
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding,
                name=self.get_name()
            )
            if self.use_bias:
                conv_out = tf.nn.bias_add(conv_out, self.b, name=self.get_name() + ConvLayer.BIAS)
            if self.f is None:
                return conv_out
            return self.f(conv_out, name=self.get_name() + ConvLayer.ACTIVATION_PREFIX)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[ConvLayer.SHAPE][0]
        kh = params[ConvLayer.SHAPE][1]
        in_f = params[ConvLayer.SHAPE][2]
        out_f = params[ConvLayer.SHAPE][3]

        stride = params[ConvLayer.STRIDE]
        padding = params[ConvLayer.PADDING]
        activation = ActivationConverter.str_to_activation(params[ConvLayer.ACTIVATION])

        init_type = params[ConvLayer.INIT_TYPE]
        use_bias = params[ConvLayer.USE_BIAS]

        return ConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f,
            stride=stride, name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ConvLayer.SHAPE: list(self.shape),
                ConvLayer.STRIDE: self.stride,
                ConvLayer.PADDING: self.padding,
                ConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                ConvLayer.USE_BIAS: self.use_bias,
                ConvLayer.INIT_TYPE: self.init_type
            }

        }


class UpConvLayer(SimpleForwardLayer):
    TYPE = 'UpConvLayer'
    SHAPE = 'shape'
    SIZE = 'size'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_BIAS = 'UpConvBias_{}x{}_in{}_out{}_id_{}'
    NAME_CONV_W = 'UpConvKernel_{}x{}_out{}_in{}_id_{}'

    def __init__(self, kw, kh, in_f, out_f, name, size=(2, 2), padding='SAME', activation=tf.nn.relu,
                 kernel_initializer=InitConvKernel.HE, use_bias=True, regularize_bias=False, W=None, b=None):
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
            Padding mode for convolution operation.
            Options: UpConvLayer.PADDING_SAME which is 'SAME' string
            or UpConvLayer.PADDING_VALID 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        """
        # Shape is different from normal convolution since it's required by
        # transposed convolution. Output feature maps go before input ones.
        self.shape = (kw, kh, out_f, in_f)
        self.size = size
        self.strides = [1, *size, 1]
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, in_f, out_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_conv = UpConvLayer.NAME_CONV_W.format(kw, kh, out_f, in_f, name)
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        regularize_params = [self.W]

        if use_bias:
            self.name_bias = UpConvLayer.NAME_BIAS.format(kw, kh, in_f, out_f, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            out_shape = X.get_shape().as_list()
            out_shape[1] *= self.size[0]
            out_shape[2] *= self.size[1]
            # out_f
            out_shape[3] = self.shape[2]
            conv_out = tf.nn.conv2d_transpose(
                X, self.W,
                output_shape=out_shape, strides=self.strides, padding=self.padding,
                name=self.get_name()
            )
            if self.use_bias:
                conv_out = tf.nn.bias_add(conv_out, self.b, name=self.get_name() + UpConvLayer.BIAS)

            if self.f is None:
                return conv_out
            return self.f(conv_out, name=self.get_name() + UpConvLayer.ACTIVATION_PREFIX)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[UpConvLayer.SHAPE][0]
        kh = params[UpConvLayer.SHAPE][1]
        in_f = params[UpConvLayer.SHAPE][3]
        out_f = params[UpConvLayer.SHAPE][2]

        padding = params[UpConvLayer.PADDING]
        size = params[UpConvLayer.SIZE]

        activation = ActivationConverter.str_to_activation(params[UpConvLayer.ACTIVATION])

        init_type = params[UpConvLayer.INIT_TYPE]
        use_bias = params[UpConvLayer.USE_BIAS]
        return UpConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, size=size,
            name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: UpConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                UpConvLayer.SHAPE: list(self.shape),
                UpConvLayer.SIZE: self.size,
                UpConvLayer.PADDING: self.padding,
                UpConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                UpConvLayer.USE_BIAS: self.use_bias,
                UpConvLayer.INIT_TYPE: self.init_type
            }
        }


class BiasLayer(SimpleForwardLayer):
    TYPE = 'BiasLayer'
    D = 'D'
    TRAINABLE = 'trainable'

    NAME_BIAS = 'BiasLayer_{}_{}'

    def __init__(self, D, name, trainable=True, regularize_bias=False, b=None):
        """
        BiasLayer adds a bias vector of dimension D to a tensor.
        If `trainable` set to False, then the bias is meant to be a constant.
        It is done this way to prevent the bias from accidential training
        if the user simply forgot to make this layer untrainable after creating a model.

        Parameters
        ----------
        D : int
            Dimension of bias vector.
        name : str
            Name of this layer.
        trainable : bool
            If true, bias will be learned in train, otherwise its will be constant variable.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with predefined value.
        """
        self.D = D
        self.name = name
        self.trainable = trainable

        if b is None:
            b = np.zeros(D)
        elif b.shape[0] != D:
            raise ValueError(f"The initial value of `b` must have the same dimension size as D={D}")

        params = []
        regularize_params = []

        self.bias_name = BiasLayer.NAME_BIAS.format(D, name)
        self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)

        if trainable:
            params = [self.b]
            if regularize_bias:
                regularize_params = [self.b]

        named_params_dict = {self.bias_name: self.b}

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            return tf.nn.bias_add(X, self.b, self.get_name())

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        D = params[BiasLayer.D]
        trainable = params[BiasLayer.TRAINABLE]
        return BiasLayer(D=D, trainable=trainable, name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: BiasLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                BiasLayer.D: self.D,
                BiasLayer.TRAINABLE: self.trainable,
            }
        }


class DepthWiseConvLayer(SimpleForwardLayer):
    TYPE = 'DepthWiseLayer'
    SHAPE = 'shape'
    STRIDE = 'stride'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'
    RATE = 'rate'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_CONV_W = 'DepthWiseConvKernel_{}x{}_in{}_out{}_id_{}'
    NAME_BIAS = 'DepthWiseConvBias_{}{}'

    def __init__(self, kw, kh, in_f, multiplier, name, stride=1, padding='SAME', rate=[1, 1],
                 kernel_initializer=InitConvKernel.HE, use_bias=True, activation=tf.nn.relu,
                 regularize_bias=False, W=None, b=None):
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
        multiplier : int
            Number of output feature maps equals `in_f`*`multiplier`.
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation.
            Options: DepthWiseConvLayer.PADDING_SAME which is 'SAME' string
            or DepthWiseConvLayer.PADDING_VALID 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        assert (len(rate) == 2)
        self.shape = (kw, kh, in_f, multiplier)
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.rate = rate
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, multiplier, in_f, kernel_initializer)
        if b is None:
            b = np.zeros(in_f * multiplier)

        self.name_conv = DepthWiseConvLayer.NAME_CONV_W.format(kw, kh, in_f, multiplier, name)
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        regularize_params = [self.W]

        if use_bias:
            self.bias_name = DepthWiseConvLayer.NAME_BIAS.format(in_f * multiplier, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)
            params += [self.b]
            named_params_dict[self.bias_name] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            conv_out = tf.nn.depthwise_conv2d(
                input=X,
                filter=self.W,
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding,
                rate=self.rate,
                name=self.get_name()
            )
            if self.use_bias:
                conv_out = tf.nn.bias_add(conv_out, self.b, name=self.get_name() + DepthWiseConvLayer.BIAS)
            if self.f is None:
                return conv_out
            return self.f(conv_out, name=self.get_name() + DepthWiseConvLayer.ACTIVATION_PREFIX)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[DepthWiseConvLayer.SHAPE][0]
        kh = params[DepthWiseConvLayer.SHAPE][1]
        in_f = params[DepthWiseConvLayer.SHAPE][2]
        multiplier = params[DepthWiseConvLayer.SHAPE][3]

        padding = params[DepthWiseConvLayer.PADDING]
        stride = params[DepthWiseConvLayer.STRIDE]

        init_type = params[DepthWiseConvLayer.INIT_TYPE]
        use_bias = params[DepthWiseConvLayer.USE_BIAS]
        rate = params[DepthWiseConvLayer.RATE]

        activation = ActivationConverter.str_to_activation(params[DepthWiseConvLayer.ACTIVATION])

        return DepthWiseConvLayer(
            kw=kw, kh=kh, in_f=in_f, multiplier=multiplier, padding=padding,
            stride=stride, activation=activation, name=name, rate=rate,
            kernel_initializer=init_type, use_bias=use_bias,
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: DepthWiseConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                DepthWiseConvLayer.SHAPE: list(self.shape),
                DepthWiseConvLayer.STRIDE: self.stride,
                DepthWiseConvLayer.PADDING: self.padding,
                DepthWiseConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                DepthWiseConvLayer.USE_BIAS: self.use_bias,
                DepthWiseConvLayer.INIT_TYPE: self.init_type,
                DepthWiseConvLayer.RATE: self.rate,
            }
        }


class SeparableConvLayer(SimpleForwardLayer):
    TYPE = 'SeparableConvLayer'
    DW_SHAPE = 'dw_shape'
    OUT_F = 'out_f'
    STRIDE = 'stride'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    DW_INIT_TYPE = 'dw_init_type'
    PW_INIT_TYPE = 'pw_init_type'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_DW = 'DWConvKernel_{}x{}_in{}_out{}_id_{}'
    NAME_PW = 'PWConvKernel_1x1_in{}_out{}_id_{}'
    NAME_BIAS = 'SeparableConvBias_{}{}'

    def __init__(self, kw, kh, in_f, out_f, multiplier, name, stride=1, padding='SAME',
                 dw_kernel_initializer=InitConvKernel.XAVIER_GAUSSIAN_INF, pw_kernel_initializer=InitConvKernel.HE,
                 use_bias=True, regularize_bias=False, activation=tf.nn.relu,
                 W_dw=None, W_pw=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of the input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int
            Number of the output feature maps after pointwise convolution,
            i.e. it is depth of the final output tensor.
        multiplier : int
            Number of output feature maps after depthwise convolution equals `in_f`*`multiplier`.
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation.
            Options: SeparableConvLayer.PADDING_SAME which is 'SAME' string
            or SeparableConvLayer.PADDING_VALID 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W_dw : numpy array
            Filter's weights. This value is used for the filter initialization.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.dw_shape = (kw, kh, in_f, multiplier)
        self.out_f = out_f
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.dw_init_type = dw_kernel_initializer
        self.pw_init_type = pw_kernel_initializer

        name = str(name)

        if W_dw is None:
            W_dw = InitConvKernel.init_by_name(kw, kh, multiplier, in_f, dw_kernel_initializer)
        if W_pw is None:
            W_pw = InitConvKernel.init_by_name(1, 1, out_f, multiplier * in_f, pw_kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_DW = SeparableConvLayer.NAME_DW.format(kw, kh, in_f, multiplier, name)
        self.name_PW = SeparableConvLayer.NAME_PW.format(in_f * multiplier, out_f, name)
        self.W_dw = tf.Variable(W_dw, name=self.name_DW)
        self.W_pw = tf.Variable(W_pw, name=self.name_PW)
        params = [self.W_dw, self.W_pw]
        named_params_dict = {
            self.name_DW: self.W_dw,
            self.name_PW: self.W_pw,
        }
        regularize_params = [self.W_dw, self.W_pw]
        if use_bias:
            self.bias_name = SeparableConvLayer.NAME_BIAS.format(out_f, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)
            params += [self.b]
            named_params_dict[self.bias_name] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            conv_out = tf.nn.separable_conv2d(
                input=X,
                depthwise_filter=self.W_dw,
                pointwise_filter=self.W_pw,
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding,
                name=self.get_name()
            )
            if self.use_bias:
                conv_out = tf.nn.bias_add(conv_out, self.b, name=self.get_name() + SeparableConvLayer.BIAS)
            if self.f is None:
                return conv_out
            return self.f(conv_out, name=self.get_name() + SeparableConvLayer.ACTIVATION_PREFIX)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[SeparableConvLayer.DW_SHAPE][0]
        kh = params[SeparableConvLayer.DW_SHAPE][1]
        in_f = params[SeparableConvLayer.DW_SHAPE][2]
        out_f = params[SeparableConvLayer.OUT_F]

        multiplier = params[SeparableConvLayer.DW_SHAPE][3]

        padding = params[SeparableConvLayer.PADDING]
        stride = params[SeparableConvLayer.STRIDE]

        dw_init_type = params[SeparableConvLayer.DW_INIT_TYPE]
        pw_init_type = params[SeparableConvLayer.PW_INIT_TYPE]
        use_bias = params[SeparableConvLayer.USE_BIAS]

        activation = ActivationConverter.str_to_activation(params[SeparableConvLayer.ACTIVATION])

        return SeparableConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, multiplier=multiplier,
            padding=padding, stride=stride, activation=activation,
            dw_kernel_initializer=dw_init_type, pw_kernel_initializer=pw_init_type,
            use_bias=use_bias, name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: SeparableConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                SeparableConvLayer.DW_SHAPE: list(self.dw_shape),
                SeparableConvLayer.OUT_F: self.out_f,
                SeparableConvLayer.STRIDE: self.stride,
                SeparableConvLayer.PADDING: self.padding,
                SeparableConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                SeparableConvLayer.USE_BIAS: self.use_bias,
                SeparableConvLayer.DW_INIT_TYPE: self.dw_init_type,
                SeparableConvLayer.PW_INIT_TYPE: self.pw_init_type
            }
        }


class DenseLayer(SimpleForwardLayer):
    TYPE = 'DenseLayer'
    INPUT_SHAPE = 'input_shape'
    OUTPUT_SHAPE = 'output_shape'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_DENSE_W = 'DenseMat_{}x{}_id_{}'
    NAME_BIAS = 'DenseBias_{}x{}_id_{}'

    def __init__(self, in_d, out_d, name, activation=tf.nn.relu, mat_initializer=InitDenseMat.HE,
                 use_bias=True, regularize_bias=False, W=None, b=None):
        """
        Paremeters
        ----------
        in_d : int
            Dimensionality of the input vector. Example: 500.
        out_d : int
            Dimensionality of the output vector. Example: 100.
        activation : TensorFlow function
            Activation function. Set to None if you don't need activation.
        W : numpy ndarray
            Used for initialization the weight matrix.
        b : numpy ndarray
            Used for initialisation the bias vector.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.input_shape = in_d
        self.output_shape = out_d
        self.f = activation
        self.use_bias = use_bias
        self.init_type = mat_initializer

        if W is None:
            W = InitDenseMat.init_by_name(in_d, out_d, mat_initializer)

        if b is None:
            b = np.zeros(out_d)

        name = str(name)
        self.name_dense = DenseLayer.NAME_DENSE_W.format(in_d, out_d, name)
        self.W = tf.Variable(W, name=self.name_dense)
        params = [self.W]
        named_params_dict = {self.name_dense: self.W}
        regularize_params = [self.W]

        if use_bias:
            self.name_bias = DenseLayer.NAME_BIAS.format(in_d, out_d, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            out = tf.matmul(X, self.W, name=self.get_name())
            if self.use_bias:
                out = tf.nn.bias_add(out, self.b, name=self.get_name() + DenseLayer.BIAS)
            if self.f is None:
                return out
            return self.f(out, name=self.get_name() + DenseLayer.ACTIVATION_PREFIX)

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        input_shape = params[DenseLayer.INPUT_SHAPE]
        output_shape = params[DenseLayer.OUTPUT_SHAPE]

        activation = ActivationConverter.str_to_activation(params[DenseLayer.ACTIVATION])

        init_type = params[DenseLayer.INIT_TYPE]
        use_bias = params[DenseLayer.USE_BIAS]

        return DenseLayer(
            in_d=input_shape, out_d=output_shape,
            activation=activation, name=name,
            mat_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: DenseLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                DenseLayer.INPUT_SHAPE: self.input_shape,
                DenseLayer.OUTPUT_SHAPE: self.output_shape,
                DenseLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                DenseLayer.USE_BIAS: self.use_bias,
                DenseLayer.INIT_TYPE: self.init_type
            }
        }


class AtrousConvLayer(SimpleForwardLayer):
    TYPE = 'AtrousConvLayer'
    SHAPE = 'shape'
    RATE = 'rate'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    PADDING_SAME = 'SAME'
    PADDING_VALID = 'VALID'

    BIAS = '_bias'
    ACTIVATION_PREFIX = '_activation'

    NAME_ATROUS_W = 'AtrousConvKernel_{}x{}_in{}_out{}_id_{}'
    NAME_BIAS = 'AtrousConvBias_{}x{}_in{}_out{}_id_{}'

    def __init__(self, kw, kh, in_f, out_f, rate, name, padding='SAME', activation=tf.nn.relu,
                 kernel_initializer=InitConvKernel.HE, use_bias=True, regularize_bias=False, W=None, b=None):
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
        rate : int
            A positive int. The stride with which we sample input values across the height and width dimensions
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation.
            Options: AtrousConvLayer.PADDING_SAME which is 'SAME' string
            or AtrousConvLayer.PADDING_VALID 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.rate = rate
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)
        self.name_conv = AtrousConvLayer.NAME_ATROUS_W.format(kw, kh, in_f, out_f, name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, out_f, in_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        regularize_params = [self.W]

        if use_bias:
            self.name_bias = AtrousConvLayer.NAME_BIAS.format(kw, kh, in_f, out_f, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            conv_out = tf.nn.atrous_conv2d(X, self.W,
                                           self.rate,
                                           self.padding,
                                           name=self.get_name()
            )
            if self.use_bias:
                conv_out = tf.nn.bias_add(conv_out, self.b, name=self.get_name() + AtrousConvLayer.BIAS)
            if self.f is None:
                return conv_out
            return self.f(conv_out, name=self.get_name() + AtrousConvLayer.ACTIVATION_PREFIX)

    def _training_forward(self, x):
        return self._forward(x, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[AtrousConvLayer.SHAPE][0]
        kh = params[AtrousConvLayer.SHAPE][1]
        in_f = params[AtrousConvLayer.SHAPE][2]
        out_f = params[AtrousConvLayer.SHAPE][3]

        rate = params[AtrousConvLayer.RATE]
        padding = params[AtrousConvLayer.PADDING]

        init_type = params[AtrousConvLayer.INIT_TYPE]
        use_bias = params[AtrousConvLayer.USE_BIAS]

        activation = ActivationConverter.str_to_activation(params[AtrousConvLayer.ACTIVATION])

        return AtrousConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, rate=rate,
            padding=padding, activation=activation,
            kernel_initializer=init_type,
            use_bias=use_bias, name=name
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: AtrousConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                AtrousConvLayer.SHAPE: list(self.shape),
                AtrousConvLayer.RATE: self.rate,
                AtrousConvLayer.PADDING: self.padding,
                AtrousConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                AtrousConvLayer.USE_BIAS: self.use_bias,
                AtrousConvLayer.INIT_TYPE: self.init_type
            }
        }


class BatchNormLayer(BatchNormBaseLayer):
    TYPE = 'BatchNormLayer'

    NAME_MEAN = 'BatchMean_{}_id_{}'
    NAME_VAR = 'BatchVar_{}_id_{}'

    def __init__(self, D, name, decay=0.9, eps=1e-4, use_gamma=True, use_beta=True, regularize_gamma=False,
                 regularize_beta=False, mean=None, var=None, gamma=None, beta=None, track_running_stats=True):
        """
        Batch Normalization Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        gamma and beta are defined by the NN, e.g. they are trainable.

        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer.
        mean : float
            Batch mean value. Used for initialization mean with pretrained value.
        var : float
            Batch variance value. Used for initialization variance with pretrained value.
        gamma : float
            Batchnorm gamma value. Used for initialization gamma with pretrained value.
        beta : float
            Batchnorm beta value. Used for initialization beta with pretrained value.
        """
        super().__init__(D=D, decay=decay, eps=eps, name=name, use_gamma=use_gamma, use_beta=use_beta,
                         regularize_gamma=regularize_gamma, regularize_beta=regularize_beta,
                         type_norm='Batch', mean=mean, var=var, gamma=gamma, beta=beta, track_running_stats=track_running_stats)

    def _init_train_params(self, data):
        if self.running_mean is None:
            self.running_mean = np.zeros(self.D)
        if self.running_variance is None:
            self.running_variance = np.ones(self.D)

        name = str(self.get_name())

        self.name_mean = BatchNormLayer.NAME_MEAN.format(self.D, name)
        self.name_var = BatchNormLayer.NAME_VAR.format(self.D, name)

        self.running_mean = tf.Variable(self.running_mean.astype(np.float32), trainable=False, name=self.name_mean)
        self._named_params_dict[self.name_mean] = self.running_mean

        self.running_variance = tf.Variable(self.running_variance.astype(np.float32), trainable=False,
                                            name=self.name_var)
        self._named_params_dict[self.name_var] = self.running_variance

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            if self._track_running_stats:
                return tf.nn.batch_normalization(
                    X,
                    self.running_mean,
                    self.running_variance,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )
            else:
                # These if statements check if we do batchnorm for convolution or dense
                if len(X.shape) == 4:
                    # conv
                    axes = [0, 1, 2]
                else:
                    # dense
                    axes = [0]

                batch_mean, batch_var = tf.nn.moments(X, axes=axes)

                return tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )

    def _training_forward(self, X):
        if not self._track_running_stats:
            return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

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
                self.eps,
                name=self.get_name()
            )

        return out

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        D = params[BatchNormLayer.D]

        decay = params[BatchNormLayer.DECAY]
        eps = params[BatchNormLayer.EPS]

        use_beta = params[BatchNormLayer.USE_BETA]
        use_gamma = params[BatchNormLayer.USE_GAMMA]

        return BatchNormLayer(D=D, name=name, decay=decay, eps=eps,
                              use_beta=use_beta, use_gamma=use_gamma)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: BatchNormLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                BatchNormLayer.D: self.D,
                BatchNormLayer.DECAY: self.decay,
                BatchNormLayer.EPS: self.eps,
                BatchNormLayer.USE_BETA: self.use_beta,
                BatchNormLayer.USE_GAMMA: self.use_gamma,
                BatchNormLayer.TRACK_RUNNING_STATS: self._track_running_stats,
            }
        }


class GroupNormLayer(BatchNormBaseLayer):
    TYPE = 'GroupNormLayer'
    G = 'G'

    NAME_MEAN = 'GroupNormMean_{}_{}_id_{}'
    NAME_VAR = 'GroupNormVar_{}_{}_id_{}'

    def __init__(self, D, name, G=32, decay=0.999, eps=1e-3, use_gamma=True, regularize_gamma=False,
                 regularize_beta=False, use_beta=True, mean=None, var=None, gamma=None,
                 beta=None, track_running_stats=True):
        """
        GroupNormLayer Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        There X (as original) have shape [N, H, W, C], but in this operation it will be [N, H, W, G, C // G].
        GroupNormLayer normilized input on N and C // G axis.
        gamma and beta are learned using gradient descent.
        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        G : int
            The number of groups that normalized. NOTICE! The number D must be divisible by G without remainder
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer.
        mean : float
            Batch mean value. Used for initialization mean with pretrained value.
        var : float
            Batch variance value. Used for initialization variance with pretrained value.
        gamma : float
            Batchnorm gamma value. Used for initialization gamma with pretrained value.
        beta : float
            Batchnorm beta value. Used for initialization beta with pretrained value.
        """
        self.G = G
        super().__init__(D=D, decay=decay, eps=eps, name=name, use_gamma=use_gamma,
                         regularize_gamma=regularize_gamma, regularize_beta=regularize_beta,
                         type_norm='GroupNorm', use_beta=use_beta, mean=mean, var=var,
                         gamma=gamma, beta=beta, track_running_stats=track_running_stats)

    def _init_train_params(self, data):
        N = data.shape[0]
        shape = data.shape
        if self.running_mean is None:
            if len(shape) == 4:
                # Conv,
                self.running_mean = np.zeros((N, 1, 1, self.G, 1))
            elif len(shape) == 2:
                # Dense
                self.running_mean = np.zeros((N, self.G, 1))

        if self.running_variance is None:
            if len(shape) == 4:
                # Conv
                self.running_variance = np.ones((N, 1, 1, self.G, 1))
            elif len(shape) == 2:
                # Dense
                self.running_variance = np.ones((N, self.G, 1))

        name = str(self.get_name())

        self.name_mean = GroupNormLayer.NAME_MEAN.format(N, self.G, name)
        self.name_var = GroupNormLayer.NAME_VAR.format(N, self.G, name)

        self.running_mean = tf.Variable(self.running_mean.astype(np.float32), trainable=False, name=self.name_mean)
        self._named_params_dict[self.name_mean] = self.running_mean

        self.running_variance = tf.Variable(self.running_variance.astype(np.float32), trainable=False,
                                            name=self.name_var)
        self._named_params_dict[self.name_var] = self.running_variance

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            # These if statements check if we do batchnorm for convolution or dense
            if len(X.shape) == 4:
                # conv
                axes = [1, 2, 4]

                N, H, W, C = X.shape
                old_shape = [N, H, W, C]
                X = tf.reshape(X, [N, H, W, self.G, C // self.G])
            else:
                # dense
                axes = [2]

                N, F = X.shape
                old_shape = [N, F]
                X = tf.reshape(X, [N, self.G, F // self.G])

            if self._track_running_stats:
                X = (X - self.running_mean) / tf.sqrt(self.running_variance + self.eps)
            else:
                # Output shape [N, 1, 1, self.G, 1] for Conv and [N, G, 1] for Dense
                batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)
                X = (X - batch_mean) / tf.sqrt(batch_var + self.eps)

            X = tf.reshape(X, old_shape)

            if self.gamma is not None:
                X *= self.gamma

            if self.beta is not None:
                X += self.beta

            return X

    def _training_forward(self, X):
        if not self._track_running_stats:
            return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

        # These if statements check if we do batchnorm for convolution or dense
        if len(X.shape) == 4:
            # conv
            axes = [1, 2, 4]

            N, H, W, C = X.shape
            old_shape = [N, H, W, C]
            X = tf.reshape(X, [N, H, W, self.G, C // self.G])
        else:
            # dense
            axes = [2]

            N, F = X.shape
            old_shape = [N, F]
            X = tf.reshape(X, [N, self.G, F // self.G])

        # Output shape [N, 1, 1, self.G, 1] for Conv and [N, G, 1] for Dense
        batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)

        update_running_mean = tf.assign(
            self.running_mean,
            self.running_mean * self.decay + batch_mean * (1 - self.decay)
        )
        update_running_variance = tf.assign(
            self.running_variance,
            self.running_variance * self.decay + batch_var * (1 - self.decay)
        )

        with tf.control_dependencies([update_running_mean, update_running_variance]):
            X = (X - batch_mean) / tf.sqrt(batch_var + self.eps)

            X = tf.reshape(X, old_shape)

            if self.gamma is not None:
                X *= self.gamma

            if self.beta is not None:
                X += self.beta

        return X

    @staticmethod
    def build(params: dict):
        G = params[GroupNormLayer.G]
        name = params[MakiRestorable.NAME]
        D = params[GroupNormLayer.D]

        decay = params[GroupNormLayer.DECAY]
        eps = params[GroupNormLayer.EPS]

        use_beta = params[GroupNormLayer.USE_BETA]
        use_gamma = params[GroupNormLayer.USE_GAMMA]

        track_running_stats = params[GroupNormLayer.TRACK_RUNNING_STATS]

        return GroupNormLayer(D=D, G=G, name=name, decay=decay, eps=eps,
                              use_beta=use_beta, use_gamma=use_gamma, track_running_stats=track_running_stats)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: GroupNormLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                GroupNormLayer.D: self.D,
                GroupNormLayer.DECAY: self.decay,
                GroupNormLayer.EPS: self.eps,
                GroupNormLayer.G: self.G,
                GroupNormLayer.USE_BETA: self.use_beta,
                GroupNormLayer.USE_GAMMA: self.use_gamma,
                GroupNormLayer.TRACK_RUNNING_STATS: self._track_running_stats,
            }
        }


class NormalizationLayer(BatchNormBaseLayer):
    TYPE = 'NormalizationLayer'

    NAME_MEAN = 'NormalizationLayerMean_{}_id_{}'
    NAME_VAR = 'NormalizationLayerVar_{}__id_{}'

    def __init__(self, D, name, decay=0.999, eps=1e-3, use_gamma=True, regularize_gamma=False,
                 regularize_beta=False, use_beta=True, mean=None, var=None, gamma=None,
                 beta=None, track_running_stats=True):
        """
        NormalizationLayer Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        There X have shape [N, H, W, C]. NormalizationLayer normilized input on N axis
        gamma and beta are learned using gradient descent.
        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer.
        mean : float
            Batch mean value. Used for initialization mean with pretrained value.
        var : float
            Batch variance value. Used for initialization variance with pretrained value.
        gamma : float
            Batchnorm gamma value. Used for initialization gamma with pretrained value.
        beta : float
            Batchnorm beta value. Used for initialization beta with pretrained value.
        """
        super().__init__(D=D, decay=decay, eps=eps, name=name, use_gamma=use_gamma, use_beta=use_beta, mean=mean,
                         regularize_gamma=regularize_gamma, regularize_beta=regularize_beta,
                         type_norm='NormalizationLayer', var=var, gamma=gamma, beta=beta, track_running_stats=track_running_stats)

    def _init_train_params(self, data):
        N = data.shape[0]
        shape = data.shape
        if self.running_mean is None:
            if len(shape) == 4:
                # Conv
                self.running_mean = np.zeros((N, 1, 1, 1))
            elif len(shape) == 2:
                # Dense
                self.running_mean = np.zeros((N, 1))

        if self.running_variance is None:
            if len(shape) == 4:
                # Conv
                self.running_variance = np.ones((N, 1, 1, 1))
            elif len(shape) == 2:
                # Dense
                self.running_variance = np.ones((N, 1))

        name = str(self.get_name())
        self.name_mean = NormalizationLayer.NAME_MEAN.format(N, name)
        self.name_var = NormalizationLayer.NAME_VAR.format(N, name)

        self.running_mean = tf.Variable(self.running_mean.astype(np.float32), trainable=False, name=self.name_mean)
        self._named_params_dict[self.name_mean] = self.running_mean

        self.running_variance = tf.Variable(self.running_variance.astype(np.float32), trainable=False,
                                            name=self.name_var)
        self._named_params_dict[self.name_var] = self.running_variance

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            if self._track_running_stats:
                return tf.nn.batch_normalization(
                    X,
                    self.running_mean,
                    self.running_variance,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )
            else:
                # These if statements check if we do batchnorm for convolution or dense
                if len(X.shape) == 4:
                    # conv
                    axes = [1, 2, 3]
                else:
                    # dense
                    axes = [1]

                # Output shape [N, 1, 1, 1] for Conv and [N, 1] for Dense
                batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)

                return tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )

    def _training_forward(self, X):
        if not self._track_running_stats:
            return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

        # These if statements check if we do batchnorm for convolution or dense
        if len(X.shape) == 4:
            # conv
            axes = [1, 2, 3]
        else:
            # dense
            axes = [1]

        # Output shape [N, 1, 1, 1] for Conv and [N, 1] for Dense
        batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)

        update_running_mean = tf.assign(
            self.running_mean,
            self.running_mean * self.decay + batch_mean * (1 - self.decay)
        )
        update_running_variance = tf.assign(
            self.running_variance,
            self.running_variance * self.decay + batch_var * (1 - self.decay)
        )

        with tf.control_dependencies([update_running_mean, update_running_variance]):
            X = tf.nn.batch_normalization(
                X,
                batch_mean,
                batch_var,
                self.beta,
                self.gamma,
                self.eps,
                name=self.get_name()
            )

        return X

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        D = params[NormalizationLayer.D]

        decay = params[NormalizationLayer.DECAY]
        eps = params[NormalizationLayer.EPS]

        use_beta = params[NormalizationLayer.USE_BETA]
        use_gamma = params[NormalizationLayer.USE_GAMMA]

        track_running_stats = params[NormalizationLayer.TRACK_RUNNING_STATS]

        return NormalizationLayer(D=D, name=name, decay=decay, eps=eps,
                              use_beta=use_beta, use_gamma=use_gamma, track_running_stats=track_running_stats)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: NormalizationLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                NormalizationLayer.D: self.D,
                NormalizationLayer.DECAY: self.decay,
                NormalizationLayer.EPS: self.eps,
                NormalizationLayer.USE_BETA: self.use_beta,
                NormalizationLayer.USE_GAMMA: self.use_gamma,
                NormalizationLayer.TRACK_RUNNING_STATS: self._track_running_stats,
            }
        }


class InstanceNormLayer(BatchNormBaseLayer):
    TYPE = 'InstanceNormLayer'

    NAME_MEAN = 'InstanceNormMean_{}_{}_id_{}'
    NAME_VAR = 'InstanceNormVar_{}_{}_id_{}'

    def __init__(self, D, name, decay=0.999, eps=1e-3, use_gamma=True, regularize_gamma=False,
                 regularize_beta=False, use_beta=True, mean=None, var=None, gamma=None,
                 beta=None, track_running_stats=True):
        """
        InstanceNormLayer Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta

        There X have shape [N, H, W, C]. InstanceNormLayer normalized input on N and C axis
        gamma and beta are learned using gradient descent.

        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer.
        mean : float
            Batch mean value. Used for initialization mean with pretrained value.
        var : float
            Batch variance value. Used for initialization variance with pretrained value.
        gamma : float
            Batchnorm gamma value. Used for initialization gamma with pretrained value.
        beta : float
            Batchnorm beta value. Used for initialization beta with pretrained value.
        """
        super().__init__(D=D, decay=decay, eps=eps, name=name, use_gamma=use_gamma, use_beta=use_beta, mean=mean,
                         regularize_gamma=regularize_gamma, regularize_beta=regularize_beta,
                         type_norm='InstanceNorm', var=var, gamma=gamma, beta=beta, track_running_stats=track_running_stats)

    def _init_train_params(self, data):
        N = data.shape[0]
        # [N H W C] shape
        shape = data.shape
        if self.running_mean is None:
            if len(shape) == 4:
                # Conv
                self.running_mean = np.zeros((N, 1, 1, shape[-1]))
            elif len(shape) == 2:
                # Dense
                self.running_mean = np.zeros((N, shape[-1]))

        if self.running_variance is None:
            if len(shape) == 4:
                # Conv
                self.running_variance = np.ones((N, 1, 1, shape[-1]))
            elif len(shape) == 2:
                # Dense
                self.running_variance = np.ones((N, shape[-1]))

        name = str(self.get_name())
        self.name_mean = InstanceNormLayer.NAME_MEAN.format(N, shape[-1], name)
        self.name_var = InstanceNormLayer.NAME_VAR.format(N, shape[-1], name)

        self.running_mean = tf.Variable(self.running_mean.astype(np.float32), trainable=False, name=self.name_mean)
        self._named_params_dict[self.name_mean] = self.running_mean

        self.running_variance = tf.Variable(self.running_variance.astype(np.float32), trainable=False,
                                            name=self.name_var)
        self._named_params_dict[self.name_var] = self.running_variance

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            if self._track_running_stats:
                return tf.nn.batch_normalization(
                    X,
                    self.running_mean,
                    self.running_variance,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )
            else:
                # These if statements check if we do batchnorm for convolution or dense
                if len(X.shape) == 4:
                    # conv
                    axes = [1, 2]
                else:
                    # dense
                    axes = [1]

                # Output shape [N, 1, 1, C] for Conv and [N, F] for Dense
                batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)

                return tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )

    def _training_forward(self, X):
        if not self._track_running_stats:
            return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

        # These if statements check if we do batchnorm for convolution or dense
        if len(X.shape) == 4:
            # conv
            axes = [1, 2]
        else:
            # dense
            axes = [1]
        with tf.name_scope(self.get_name() + MakiRestorable.TRAINING_MODE):
            # Output shape [N, 1, 1, C] for Conv and [N, F] for Dense
            batch_mean, batch_var = tf.nn.moments(X, axes=axes, keep_dims=True)

            update_running_mean = tf.assign(
                self.running_mean,
                self.running_mean * self.decay + batch_mean * (1 - self.decay)
            )
            update_running_variance = tf.assign(
                self.running_variance,
                self.running_variance * self.decay + batch_var * (1 - self.decay)
            )

            with tf.control_dependencies([update_running_mean, update_running_variance]):
                X = tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    self.eps,
                    name=self.get_name()
                )

            return X

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        D = params[InstanceNormLayer.D]

        decay = params[InstanceNormLayer.DECAY]
        eps = params[InstanceNormLayer.EPS]

        use_beta = params[InstanceNormLayer.USE_BETA]
        use_gamma = params[InstanceNormLayer.USE_GAMMA]

        track_running_stats = params[InstanceNormLayer.TRACK_RUNNING_STATS]

        return InstanceNormLayer(D=D, name=name, decay=decay, eps=eps,
                              use_beta=use_beta, use_gamma=use_gamma, track_running_stats=track_running_stats)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: InstanceNormLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                InstanceNormLayer.D: self.D,
                InstanceNormLayer.DECAY: self.decay,
                InstanceNormLayer.EPS: self.eps,
                InstanceNormLayer.USE_BETA: self.use_beta,
                InstanceNormLayer.USE_GAMMA: self.use_gamma,
                InstanceNormLayer.TRACK_RUNNING_STATS: self._track_running_stats,
            }
        }


class ScaleLayer(SimpleForwardLayer):
    TYPE = 'ScaleLayer'
    INIT_VALUE = 'init_value'

    NAME_SCALE = 'ScaleValue_{}'

    def __init__(self, init_value, name, regularize_scale=True):
        """
        ScaleLayer is used to multiply input MakiTensor on `init_value`, which is trainable variable.

        Parameters
        ----------
        init_value : int
            The initial value which need to multiply by input.
        name : str
            Name of this layer.
        """
        self.init_value = init_value
        self.name_scale = ScaleLayer.NAME_SCALE.format(name)

        regularize_params = []

        self.scale = tf.Variable(init_value, name=self.name_scale, dtype=tf.float32)
        if regularize_scale:
            regularize_params = [self.scale]

        super().__init__(name, params=[self.scale],
                         regularize_params=regularize_params,
                         named_params_dict={self.name_scale: self.scale}
        )

    def _forward(self, X, computation_mode=MakiRestorable.INFERENCE_MODE):
        with tf.name_scope(computation_mode + self.get_name()):
            return tf.math.multiply(X, self.scale, name=self.get_name())

    def _training_forward(self, X):
        return self._forward(X, computation_mode=MakiRestorable.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        init_value = params[ScaleLayer.INIT_VALUE]
        return ScaleLayer(init_value=init_value, name=name)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ScaleLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self.get_name(),
                ScaleLayer.INIT_VALUE: self.init_value
            }
        }


class TrainableLayerAddress:

    ADDRESS_TO_CLASSES = {
        ConvLayer.TYPE: ConvLayer,
        UpConvLayer.TYPE: UpConvLayer,
        AtrousConvLayer.TYPE: AtrousConvLayer,
        DepthWiseConvLayer.TYPE: DepthWiseConvLayer,
        SeparableConvLayer.TYPE: SeparableConvLayer,

        BiasLayer.TYPE: BiasLayer,
        DenseLayer.TYPE: DenseLayer,

        BatchNormLayer.TYPE: BatchNormLayer,
        GroupNormLayer.TYPE: GroupNormLayer,
        NormalizationLayer.TYPE: NormalizationLayer,
        InstanceNormLayer.TYPE: InstanceNormLayer,

        ScaleLayer.TYPE: ScaleLayer,
    }


from makiflow.base.maki_entities.maki_builder import MakiBuilder

MakiBuilder.register_layers(TrainableLayerAddress.ADDRESS_TO_CLASSES)

del MakiBuilder
