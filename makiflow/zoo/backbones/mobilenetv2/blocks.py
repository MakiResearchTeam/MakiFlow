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


import tensorflow as tf

from .utils import make_divisible
from makiflow.layers import *
import makiflow as mf


def inverted_res_block(
        x,
        expansion,
        alpha,
        block_id,
        out_f=None,
        in_f=None,
        stride=1,
        use_skip_connection=True,
        use_expand=False,
        activation=tf.nn.relu6,
        use_bias=False,
        bn_params={}) -> mf.core.MakiTensor:
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    expansion : int
        Magnification multiplier of feature maps.
    alpha : int
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, default number of filters from the paper are used at each layer.
    block_id : int
        Number of block (used in name of layers).
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    out_f : int
        Number of output feature maps. By default None (shape will same as `in_f`).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu6.
    use_bias : bool
        Use bias on layers or not.
    use_skip_connection : bool
        If true, sum input and output (if they are equal).
    use_expand : bool
        If true, input feature maps `in_f` will be expand to `expansion` * `in_f`.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor
    pointwise_f : int
        Output number of feature maps
    """
    inputs = x

    if in_f is None:
        in_f = x.get_shape()[-1]

    pointwise_conv_filters = int(out_f * alpha)
    pointwise_f = make_divisible(pointwise_conv_filters, 8)

    prefix = f'expanded_conv_{block_id}/'

    if use_expand:
        # Expand
        exp_f = expansion * in_f

        x = ConvLayer(kw=1,
                      kh=1,
                      in_f=in_f,
                      out_f=exp_f,
                      name=prefix + 'expand/weights',
                      stride=1,
                      use_bias=use_bias,
                      padding='SAME',
                      activation=None,
                      )(x)

        x = BatchNormLayer(D=exp_f, name=prefix + 'expand/BatchNorm', **bn_params)(x)

        x = ActivationLayer(activation=activation, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv/'
        exp_f = in_f

    # Depthwise

    x = DepthWiseConvLayer(kw=3,
                           kh=3,
                           in_f=exp_f,
                           multiplier=1,
                           activation=None,
                           stride=stride,
                           padding='SAME',
                           use_bias=use_bias,
                           name=prefix + 'depthwise/depthwise_weights',
                           )(x)

    x = BatchNormLayer(D=exp_f, name=prefix + 'depthwise/BatchNorm', **bn_params)(x)

    x = ActivationLayer(activation=activation, name=prefix + 'depthsiwe_relu')(x)

    # Project
    x = ConvLayer(kw=1,
                  kh=1,
                  in_f=exp_f,
                  out_f=pointwise_f,
                  stride=1,
                  padding='SAME',
                  use_bias=use_bias,
                  activation=None,
                  name=prefix + 'project/weights'
                  )(x)

    x = BatchNormLayer(D=pointwise_f, name=prefix + 'project/BatchNorm', **bn_params)(x)

    if use_skip_connection:
        return SumLayer(name=prefix + 'add')([inputs, x])
    else:
        return x
