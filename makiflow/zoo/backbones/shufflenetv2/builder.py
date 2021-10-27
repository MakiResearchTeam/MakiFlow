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

from makiflow.layers.utils import InitConvKernel
from makiflow.layers import *
import tensorflow as tf

from .blocks import ShuffleNetSpatialDownUnit, ShuffleNetBasicUnitBlock


def build_ShuffleNetV2(
    input_shape=None,
    model_config=[(116, 4), (232, 8), (464, 4), 1024],
    shuffle_group=2,
    include_top=False,
    num_classes=1000,
    use_bias=False,
    stride_list=(2, 2, 2, 2, 2),
    activation=tf.nn.relu,
    create_model=False,
    name_model='MakiClassificator',
    kernel_initializer=InitConvKernel.HE,
    input_tensor=None):
    """
    Build ResNet version 1 with certain parameters

    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    model_config : list
        [(out_channel, repeat_times), (out_channel, repeat_times), ...]
    shuffle_group : int
        Number of feature that need to shuffle,
        For more information, please refer to: https://arxiv.org/pdf/1707.01083.pdf
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
    num_classes : int
        Number of classes that you need to classify
    use_bias : bool
        If true, when on layers used bias operation.
    stride_list : list
        The list of strides to each layer that apply stride (with length 5),
        By default each layer will be apply stride 2
    activation : tf object
        Activation on every convolution layer.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.

    Returns
    ---------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    if input_tensor is None:
        in_x = InputLayer(input_shape=input_shape, name='Input')
    elif input_tensor is not None:
        in_x = input_tensor

    x = ConvLayer(
        kw=3, kh=3, in_f=in_x.get_shape()[-1], out_f=24, kernel_initializer=kernel_initializer,
        use_bias=use_bias, activation=activation, stride=stride_list[0], name='conv1'
    )(in_x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=f'bn_1')(x)
    x = ActivationLayer(activation=activation, name=f'activation_1')(x)
    x = MaxPoolLayer(name='maxpool1', ksize=[1, 3, 3, 1], strides=[1, stride_list[1], stride_list[1], 1])(x)

    for idx, (block, stride_single) in enumerate(zip(model_config[:-1], stride_list[2:])):
        out_channel, repeat = block

        # First block is downsampling
        x = ShuffleNetSpatialDownUnit(
            x, out_channel, f"{idx}_block_down_shufflenet_",
            shuffle_group=shuffle_group, stride=stride_single,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias, activation=activation
        )

        # Rest blocks
        for i in range(repeat - 1):
            x = ShuffleNetBasicUnitBlock(
                x=x, out_f=out_channel, stage=f"{idx}_block_num_{i}_shufflenet_",
                shuffle_group=shuffle_group, use_bias=use_bias, activation=activation,
                kernel_initializer=kernel_initializer
            )

    x = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=model_config[-1],
        kernel_initializer=kernel_initializer, use_bias=use_bias,
        activation=activation, name='final'
    )(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=f'final_bn')(x)
    x = ActivationLayer(activation=activation, name=f'final_activation')(x)

    return in_x, x
