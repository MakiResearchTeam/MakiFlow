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
from .utils import make_divisible

import makiflow as mf
from makiflow.layers import *
from makiflow.layers.initializers import He

HE_INIT = str(He)

NAME_EXPANDED_CONV = "expanded_conv"
ZERO_EXPANDED_CONV = NAME_EXPANDED_CONV + "/"
PREFIX = NAME_EXPANDED_CONV + "_{}/"

# Expand names
NAME_EXPAND = "{}expand/weights"
NAME_EXPAND_BN = "{}expand/BatchNorm"
NAME_EXPAND_ACT = "{}expand_relu"

# Depthwise names
NAME_DEPTHWISE = "{}depthwise/depthwise_weights"
NAME_DEPTHWISE_BN = "{}depthwise/BatchNorm"
NAME_DEPTHWISE_ACT = "{}depthsiwe_relu"

# Pointwise names
NAME_POINTWISE = "{}project/weights"
NAME_POINTWISE_BN = "{}project/BatchNorm"
NAME_FINAL_ADD = "{}add"


def MobileNetV2InvertedResBlock(
        x: mf.MakiTensor,
        expansion: int,
        alpha: float,
        block_id: int,
        out_f=None,
        in_f=None,
        stride=1,
        use_skip_connection=True,
        use_expand=True,
        activation=tf.nn.relu6,
        use_bias=False,
        kernel_initializer=HE_INIT,
        bn_params={}) -> mf.MakiTensor:
    """
    Parameters
    ----------
    x : mf.MakiTensor
        Input mf.MakiTensor.
    expansion : int
        Magnification multiplier of feature maps.
        In paper, usually number 6 is used
    alpha : float
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, same number of filters as in input.
        By default alpha=1 is used in paper in most cases
    block_id : int
        Number of block (used in name of layers).
    in_f : int
        Number of input feature maps.
        By default equal to None, i.e. `in_f` will be taken from input tensor.
    out_f : int
        Number of output feature maps. By default None (shape will same as `in_f`).
    stride : int
        Stride for convolution (used in depthwise convolution)
    activation : tensorflow function
        The function of activation, by default tf.nn.relu6.
    use_bias : bool
        Use bias on layers or not.
    use_skip_connection : bool
        If true, sum input and output (if they are equal).
    use_expand : bool
        If true, input feature maps `in_f` will be expand to `expansion` * `in_f`.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.initializers,
        By default He initialization are used
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    mf.MakiTensor
        Output mf.MakiTensor

    """
    # Save for skip-connection (sum) operation
    inputs = x

    if in_f is None:
        in_f = x.shape()[-1]

    # Calculate output number of f. for last ConvLayer,
    # this number should be divisible by 8 (by default)
    pointwise_f = make_divisible(int(out_f*alpha))

    prefix = PREFIX.format(str(block_id))

    # Standard scheme: expand -> depthwise -> pointwise
    if use_expand:
        # Expand stage, expand input f according to `expansion` value
        x = ConvLayer(
            kw=1,
            kh=1,
            in_f=in_f,
            out_f=int(expansion * in_f),
            name=NAME_EXPAND.format(prefix),
            use_bias=use_bias,
            activation=None,
            kernel_initializer=kernel_initializer
        )(x)

        x = BatchNormLayer(D=x.shape()[-1], name=NAME_EXPAND_BN.format(prefix), **bn_params)(x)

        x = ActivationLayer(activation=activation, name=NAME_EXPAND_ACT.format(prefix))(x)
    else:
        # Expand layer is not used in first block in model architecture MobileNetV2
        prefix = ZERO_EXPANDED_CONV

    # Depthwise stage
    x = DepthWiseConvLayer(
        kw=3,
        kh=3,
        in_f=x.shape()[-1],
        multiplier=1,
        activation=None,
        stride=stride,
        use_bias=use_bias,
        name=NAME_DEPTHWISE.format(prefix),
        kernel_initializer=kernel_initializer
    )(x)

    x = BatchNormLayer(D=x.shape()[-1], name=NAME_DEPTHWISE_BN.format(prefix), **bn_params)(x)
    x = ActivationLayer(activation=activation, name=NAME_DEPTHWISE_ACT.format(prefix))(x)

    # Pointwise (Project) to certain size (input number of the f)
    x = ConvLayer(
        kw=1,
        kh=1,
        in_f=x.shape()[-1],
        out_f=pointwise_f,
        use_bias=use_bias,
        activation=None,
        name=NAME_POINTWISE.format(prefix),
        kernel_initializer=kernel_initializer
    )(x)

    x = BatchNormLayer(D=x.shape()[-1], name=NAME_POINTWISE_BN.format(prefix), **bn_params)(x)

    if use_skip_connection:
        if x.shape()[-1] != inputs.shape()[-1]:
            # Write warning and returns `x` tensor
            if alpha == 1.0:
                print(
                    f'WARNING! SumLayer operation | \n'
                    f'In block {block_id} input and output features have different size.\n'
                    f'Skip connection will be not applyed for this block.'
                )
            else:
                print(
                    f'WARNING! SumLayer operation | \n'
                    f'In block {block_id} input and output features have different size.\n'
                    f'Skip connection will be not applied for this block.\n'
                    f'Reason: Alpha={alpha} i.e. your number of features will be decreasing or increasing '
                    f'thats why input and output features have different size.\n'
                    f'You can ignore this warning if its true.\n'
                )
            return x
        # if everything is OK - apply sum operation
        return SumLayer(name=NAME_FINAL_ADD.format(prefix))([inputs, x])

    return x

