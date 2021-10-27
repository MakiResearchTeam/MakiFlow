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

import tensorflow as tf
from .blocks import MobileNetV2InvertedResBlock, HE_INIT
from .utils import make_divisible, get_batchnorm_params

import makiflow as mf
from makiflow.layers import *
from makiflow import Model

FINAL_NUM_F = 1280
STRIDE_lIST_SIZE = 5
# TODO: Change string names into constants, like in blocks.py file
# TODO: Create some sort of loop which creates MobileNetV2InvertedResBlock by certain config

# TODO: Is this bug or super-hack? Then alpha != 1.0, we can not use skip connection
#       inside blocks, do not how the original mobilenet v2 was implemented for this case,
#       because the source code is written using tf.slim
#       which is ugly and its pain in ass to decode this magic


def build_MobileNetV2(
        in_x: mf.MakiTensor,
        use_bias=False,
        activation=tf.nn.relu6,
        alpha=1.0,
        expansion=6,
        stride_list=(2, 2, 2, 2, 2),
        bn_params=None,
        include_top=False,
        num_classes=1000,
        create_model=False,
        kernel_initializer=HE_INIT,
        name_model='MakiClassificator'):
    """
    Parameters
    ----------
    in_x : mf.MakiTensor
        A tensor that will be fed into the model as input tensor.
    use_bias : bool
        Use bias on layers or not
    activation : tensorflow function
        The function of activation, by default tf.nn.relu6
    alpha : float
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, same number of filters as in input.
        By default alpha=1 is used in paper in most cases
    expansion : int
        Magnification multiplier of feature maps.
    stride_list : list
        The list of strides to each layer that apply stride (with length 5),
        By default each layer will be apply stride 2
    bn_params : dict
        Parameters for BatchNormLayer.
        If equal to None then all parameters will have default valued taken from utils
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes
    num_classes : int
        Number of classes that you need to classify
    create_model : bool
        Return build classification model, otherwise return input mf.MakiTensor and output mf.MakiTensor
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    name_model : str
        Name of model, if it will be created

    Returns
    ---------
    output : mf.MakiTensor
        Output MakiTensor. if `create_model` is False
    model : mf.Model
        MakiFlow model. if `create_model` is True

    """
    if bn_params is None:
        bn_params = get_batchnorm_params()

    # Make sure, that `32 * alpha` is divided by 8, its important
    first_filt = make_divisible(32 * alpha)

    if len(stride_list) != STRIDE_lIST_SIZE:
        raise ValueError(
            f"Wrong size of `stride_list`, it must be {STRIDE_lIST_SIZE}, but {len(stride_list)} was received."
        )

    if not isinstance(in_x, mf.MakiTensor):
        raise ValueError(
            f"Wrong type of `in_x`, must be mf.MakiTensor, but {type(in_x)} was received. "
        )
    input_shape = in_x.shape

    x = ConvLayer(
        kw=3,
        kh=3,
        in_f=input_shape[-1],
        out_f=first_filt,
        stride=stride_list[0],
        activation=None,
        use_bias=use_bias,
        name='Conv/weights',
        kernel_initializer=kernel_initializer,
    )(in_x)

    x = BatchNormLayer(D=first_filt, name='Conv/BatchNorm', **bn_params)(x)
    x = ActivationLayer(activation=activation, name='Conv_relu')(x)
    # Do not use expansion in first block, save for skip connection
    x = MobileNetV2InvertedResBlock(
        x=x, out_f=16, alpha=alpha,
        expansion=1, block_id=0,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_expand=False, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=24, alpha=alpha, stride=stride_list[1],
        expansion=expansion, block_id=1,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=24, alpha=alpha,
        expansion=expansion, block_id=2,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=32, alpha=alpha,
        stride=stride_list[2], expansion=expansion, block_id=3,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=32, alpha=alpha,
        expansion=expansion, block_id=4,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=32, alpha=alpha,
        expansion=expansion, block_id=5,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=64, alpha=alpha,
        stride=stride_list[3], expansion=expansion, block_id=6,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=64, alpha=alpha,
        expansion=expansion, block_id=7,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=64, alpha=alpha,
        expansion=expansion, block_id=8,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x,out_f=64, alpha=alpha,
        expansion=expansion, block_id=9,
        use_bias=use_bias, activation=activation, bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=96, alpha=alpha,
        expansion=expansion, block_id=10,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=96, alpha=alpha,
        expansion=expansion, block_id=11,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=96, alpha=alpha,
        expansion=expansion, block_id=12,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=160, alpha=alpha, stride=stride_list[4],
        expansion=expansion, block_id=13,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=160, alpha=alpha,
        expansion=expansion, block_id=14,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=160, alpha=alpha,
        expansion=expansion, block_id=15,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params,
        kernel_initializer=kernel_initializer
    )

    x = MobileNetV2InvertedResBlock(
        x=x, out_f=320, alpha=alpha,
        expansion=expansion, block_id=16,
        use_bias=use_bias, activation=activation,
        bn_params=bn_params, use_skip_connection=False,
        kernel_initializer=kernel_initializer
    )
    # Final pointwise
    if alpha > 1.0:
        # Make sure its divided by 8
        last_block_filters = make_divisible(FINAL_NUM_F * alpha)
    else:
        last_block_filters = FINAL_NUM_F

    x = ConvLayer(
        kh=1,
        kw=1,
        in_f=x.shape()[-1],
        out_f=last_block_filters,
        activation=None,
        use_bias=use_bias,
        name='Conv_1/weights',
        kernel_initializer=kernel_initializer
    )(x)

    x = BatchNormLayer(D=last_block_filters, name='Conv_1/BatchNorm', **bn_params)(x)
    x = ActivationLayer(activation=activation, name='out_relu')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='global_avg')(x)
        x = ReshapeLayer(new_shape=[1,1, x.shape()[-1]], name='resh')(x)
        x = ConvLayer(kw=1, kh=1, in_f=x.shape()[-1], out_f=num_classes, name='prediction')(x)
        output = ReshapeLayer(new_shape=[num_classes], name='endo')(x)
    else:
        output = x

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)

    return output

