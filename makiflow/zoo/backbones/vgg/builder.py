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
from makiflow.models import Classificator
import tensorflow as tf

from .blocks import VGGBlock, NONE


def build_VGG(
    input_shape=None,
    input_tensor=None,
    use_bias=False,
    activation=tf.nn.relu,
    repetition=3,
    number_of_blocks=5,
    init_fm=64,
    pooling_type='max_pool',
    is_use_pool_list=(True, True, True, True, True),
    include_top=False,
    num_classes=1000,
    create_model=False,
    pool_params=None,
    kernel_initializer=InitConvKernel.HE,
    name_model='MakiClassificator'):
    """
    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.
    use_bias : bool
        If true, when on layers used bias operation.
    activation : tf object
        Activation on every convolution layer.
    repetition : int
        Number of repetition convolution per block, usually 3 for VGG16, 4 for vgg 19.
    number_of_blocks : int
        Number of blocks of `repetition`.
    init_fm : int
        Initial number of feature maps.
    pooling_type : str
        What type of pooling are will be used.
        'max_pool' - for max pooling.
        'avg_pool' - for average pooling.
        'none' or any other strings - the operation pooling will not be used.
    is_use_pool_list : list
        The list of boolean variables to each layer,
        if equal to True then pool operation will be applied to certain block
        By default each layer will be apply stride 2
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes
    num_classes : int
        Number of classes that you need to classify
    create_model : bool
        Return build classification model, otherwise return input mf.MakiTensor and output mf.MakiTensor
    pool_params : dict
        Parameters for pool operation, by default equal to None, i. e. parameters will be taken from utilss
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    name_model : str
        Name of model, if it will be created

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
    assert len(is_use_pool_list) == number_of_blocks

    if repetition <= 0:
        raise TypeError('repetition should have type int and be more than 0')

    if input_tensor is None and input_shape is not None:
        in_x = InputLayer(input_shape=input_shape, name='Input')
    elif input_tensor is not None:
        in_x = input_tensor
    else:
        raise ValueError(
            "Wrong input `input_tensor` or `input_shape`"
        )

    for i in range(1, number_of_blocks):
        if is_use_pool_list[i-1]:
            type_pool = pooling_type
        else:
            type_pool = NONE

        # First block
        if i == 1:
            x = VGGBlock(
                x=in_x, num_block=str(i), n=2,
                out_f=init_fm, pooling_type=type_pool,
                use_bias=use_bias, activation=activation,
                kernel_initializer=kernel_initializer, pool_params=pool_params
            )
        # Second block
        elif i == 2:
            x = VGGBlock(
                x=x, num_block=str(i), n=2,
                pooling_type=type_pool,
                use_bias=use_bias, activation=activation,
                kernel_initializer=kernel_initializer, pool_params=pool_params
            )
        else:
            x = VGGBlock(
                x=x, num_block=str(i), n=repetition,
                pooling_type=type_pool,
                use_bias=use_bias, activation=activation,
                kernel_initializer=kernel_initializer, pool_params=pool_params
            )

    # Last block
    x = VGGBlock(
        x, out_f=x.get_shape()[-1], num_block=str(number_of_blocks),
        n=repetition, pooling_type=pooling_type if is_use_pool_list[-1] else NONE,
        use_bias=use_bias, activation=activation,
        kernel_initializer=kernel_initializer, pool_params=pool_params
    )

    if include_top:
        x = FlattenLayer(name='flatten')(x)
        x = DenseLayer(in_d=x.get_shape()[-1], out_d=4096, name='fc6')(x)
        x = DenseLayer(in_d=4096, out_d=4096, name='fc7')(x)
        output = DenseLayer(in_d=4096, out_d=num_classes, activation=None, name='fc8')(x)

        if create_model:
            return Classificator(in_x, output, name=name_model)
    else:
        output = x

    return in_x, output
