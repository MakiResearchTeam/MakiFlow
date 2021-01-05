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
from makiflow.core import MakiLayer, MakiBuilder
from .utils import InitConvKernel
from .activation_converter import ActivationConverter
import tensorflow as tf
import numpy as np


class FourierConvLayer(MakiLayer):
    TYPE = 'FourierConvLayer'
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
            Options: FourierConvLayer.PADDING_SAME which is 'SAME' string
            or FourierConvLayer.PADDING_VALID 'VALID' (case sensitive).
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

        self.name_conv = self.NAME_CONV_W.format(kw, kh, in_f, out_f, name)
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        regularize_params = [self.W]
        if use_bias:
            self.name_bias = self.NAME_BIAS.format(kw, kh, in_f, out_f, name)
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def forward(self, X, computation_mode=MakiLayer.INFERENCE_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                X = tf.fft(tf.cast(X, 'complex64'))
                X = tf.real(X)
                conv_out = tf.nn.conv2d(
                    X, self.W,
                    strides=[1, self.stride, self.stride, 1],
                    padding=self.padding,
                    name=self.get_name()
                )
                conv_out = tf.ifft(tf.cast(conv_out, 'complex64'))
                conv_out = tf.cast(tf.real(conv_out), 'float32')
                if self.f is None:
                    return conv_out
                return self.f(conv_out, name=self.get_name() + FourierConvLayer.ACTIVATION_PREFIX)

    def training_forward(self, X):
        return self.forward(X, computation_mode=MakiLayer.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        name = params[MakiLayer.NAME]

        kw = params[FourierConvLayer.SHAPE][0]
        kh = params[FourierConvLayer.SHAPE][1]
        in_f = params[FourierConvLayer.SHAPE][2]
        out_f = params[FourierConvLayer.SHAPE][3]

        stride = params[FourierConvLayer.STRIDE]
        padding = params[FourierConvLayer.PADDING]
        activation = ActivationConverter.str_to_activation(params[FourierConvLayer.ACTIVATION])

        init_type = params[FourierConvLayer.INIT_TYPE]
        use_bias = params[FourierConvLayer.USE_BIAS]

        return FourierConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f,
            stride=stride, name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiLayer.FIELD_TYPE: FourierConvLayer.TYPE,
            MakiLayer.PARAMS: {
                MakiLayer.NAME: self.get_name(),
                FourierConvLayer.SHAPE: list(self.shape),
                FourierConvLayer.STRIDE: self.stride,
                FourierConvLayer.PADDING: self.padding,
                FourierConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                FourierConvLayer.USE_BIAS: self.use_bias,
                FourierConvLayer.INIT_TYPE: self.init_type
            }

        }


MakiBuilder.register_layers({FourierConvLayer.TYPE, FourierConvLayer})
