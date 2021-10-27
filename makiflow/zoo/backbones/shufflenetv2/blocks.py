# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiZoo.
#
# MakiZoo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiZoo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from makiflow.layers import *
from makiflow.layers.utils import InitConvKernel
from makiflow.core import MakiTensor

import tensorflow as tf


def ShuffleNetBasicUnitBlock(
        x: MakiTensor,
        out_f: int,
        stage: str,
        shuffle_group=2,
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create basic unit of ShuffleNetV2.
    You can see more detail image in original paper: https://arxiv.org/pdf/1807.11164.pdf

    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    out_f : int
        Output number of feature maps
    stage : str
        Prefix to all layers, used only for layer names
    shuffle_group : int
        Number of feature that need to shuffle,
        For more information, please refer to: https://arxiv.org/pdf/1707.01083.pdf
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    MakiTensor
        Output MakiTensor of this block

    """
    if x.get_shape()[-1] % 2 != 0:
        raise ValueError(
            "Error!!!\n"
            "In ShuffleNetBasicUnitBlock input number of features must be divided by 2 without reminder."
        )
    stage = str(stage)

    # Split input tensor into 2 parts
    x1, x2 = ChannelSplitLayer(num_or_size_splits=2, axis=3, name=stage + '_split')(x)

    # Main branch
    x = ConvLayer(
        kw=1, kh=1, in_f=x1.get_shape()[-1], out_f=out_f // 2, kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + '/mb/conv1'
    )(x1)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'/mb/bn_1')(x)
    x = ActivationLayer(activation=activation, name=stage + f'/mb/activation_1')(x)

    x = DepthWiseConvLayer(
        kw=3, kh=3, in_f=x.get_shape()[-1], multiplier=1, kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + f'/mb/conv_2'
    )(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'/mb/bn_2')(x)

    x = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=x.get_shape()[-1], kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + '/mb/conv3'
    )(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'/mb/bn_3')(x)
    x = ActivationLayer(activation=activation, name=stage + f'/mb/activation_3')(x)

    # Connect reminder x1 and main branch
    x = ConcatLayer(name=stage + '/concat_f')([x2, x])
    x = ChannelShuffleLayer(num_groups=shuffle_group, name=stage + '/shuffle_f')(x)
    return x


def ShuffleNetSpatialDownUnit(
        x: MakiTensor,
        out_f: int,
        stage: str,
        shuffle_group=2,
        stride=2,
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create spatial unit of ShuffleNetV2.
    This layers usually used to reduce image size (i.e. with `stride`=2)
    You can see more detail image in original paper: https://arxiv.org/pdf/1807.11164.pdf

    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor
    out_f : int
        Output number of feature maps
    stage : str
        Prefix to all layers, used only for layer names
    shuffle_group : int
        Number of feature that need to shuffle,
        For more information, please refer to: https://arxiv.org/pdf/1707.01083.pdf
    stride : int
        Stride for this block, by defualt equal to 2
    activation : tensorflow function
        The function of activation, by default tf.nn.relu
    use_bias : bool
        Use bias on layers or not
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    MakiTensor
        Output MakiTensor of this block

    """

    # First branch
    # Conv(1x1) -> bn -> relu --> DepthWiseConv(3x3, s=2) -> bn --> Conv(1x1) -> bn -> relu
    fx = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=out_f // 2, kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + '/fx/conv1'
    )(x)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'/fx/bn_1')(fx)
    fx = ActivationLayer(activation=activation, name=stage + f'/fx/activation_1')(fx)

    fx = DepthWiseConvLayer(
        kw=3, kh=3, in_f=fx.get_shape()[-1], multiplier=1, kernel_initializer=kernel_initializer,
        use_bias=use_bias, stride=stride, activation=None, name=stage + f'/fx/conv_2'
    )(fx)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'/fx/bn_2')(fx)

    fx = ConvLayer(
        kw=1, kh=1, in_f=fx.get_shape()[-1], out_f=fx.get_shape()[-1], kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + '/fx/conv3'
    )(fx)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'/fx/bn_3')(fx)
    fx = ActivationLayer(activation=activation, name=stage + f'/fx/activation_3')(fx)

    # Second branch
    # DepthWiseConv(3x3, s=2) -> bn --> Conv(1x1) -> bn -> relu
    sx = DepthWiseConvLayer(
        kw=3, kh=3, in_f=x.get_shape()[-1], multiplier=1, kernel_initializer=kernel_initializer,
        use_bias=use_bias, stride=stride, activation=None, name=stage + f'/sx/conv_1'
    )(x)
    sx = BatchNormLayer(D=sx.get_shape()[-1], name=stage + f'/sx/bn_1')(sx)

    sx = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=out_f // 2, kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=None, name=stage + '/sx/conv2'
    )(sx)
    sx = BatchNormLayer(D=sx.get_shape()[-1], name=stage + f'/sx/bn_2')(sx)
    sx = ActivationLayer(activation=activation, name=stage + f'/sx/activation_2')(sx)

    # Connect two branches and shuffle
    x = ConcatLayer(name=stage + '/concat_f')([fx, sx])
    x = ChannelShuffleLayer(num_groups=shuffle_group, name=stage + '/shuffle_f')(x)
    return x

