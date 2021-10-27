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
from .utils import get_batchnorm_params

import makiflow as mf
from makiflow.layers import *
from makiflow.layers.initializers import He

HE_INIT = str(He)

# ResNet with pointwise operation (i.e. kernel with size 1x1)
PREFIX_NAME_BLOCK = "block{}/unit_{}"
PREFIX_NAME_SHORTCUT = "{}/bottleneck_v1/shortcut/{}"
PREFIX_NAME_LAYER = "{}/bottleneck_v1/conv{}/{}"

BATCH_NORM = "BatchNorm"
ACTIV = "activ"
WEIGHTS = "weights"
SUM_OPERATION = '/sum_operation'

# Resnet w/o pointwise (WOP) operation
PREFIX_NAME_BLOCK_WOP = "stage{}_unit{}_"
BN = "{}bn{}"
CONV = "{}conv{}"
ACTIVATION = "{}activation{}"
ZERO_PADDING = "{}zero_padding{}"
SCIP_BRANCH = "{}sc/conv"


def ResNetIdentityBlockV1(
        x: mf.MakiTensor,
        block_id: int,
        unit_id: int,
        num_block: int,
        in_f=None,
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=HE_INIT,
        bn_params=None) -> mf.MakiTensor:
    """
    Create ResNet block with skip connection,
    This type of block are presented in first paper about ResNet (i.e. v1)

    Parameters
    ----------
    x : mf.MakiTensor
        Input mf.MakiTensor.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : mf.MakiTensor
        Output mf.MakiTensor.

    """
    if bn_params is None:
        bn_params = get_batchnorm_params()

    prefix_name = PREFIX_NAME_BLOCK.format(block_id, unit_id)

    if in_f is None:
        in_f = x.shape()[-1]

    reduction = int(in_f / 4)

    mx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 1, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(x)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 1, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 1, ACTIV))(mx)

    mx = ConvLayer(
        kw=3, kh=3, in_f=reduction, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 2, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(mx)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 2, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 2, ACTIV))(mx)

    mx = ConvLayer(
        kw=1, kh=1, in_f=reduction, out_f=in_f, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 3, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(mx)
    mx = BatchNormLayer(D=in_f, name=PREFIX_NAME_LAYER.format(prefix_name, 3, BATCH_NORM), **bn_params)(mx)

    x = SumLayer(name=prefix_name + str(num_block) + SUM_OPERATION)([mx, x])

    return x


def ResNetConvBlockV1(
        x : mf.MakiTensor,
        block_id: int,
        unit_id: int,
        num_block: int,
        activation=tf.nn.relu,
        use_bias=False,
        stride=2,
        out_f=None,
        in_f=None,
        reduction=None,
        kernel_initializer=HE_INIT,
        bn_params=None):
    """
    Create ResNet block with skip connection using certain `stride`,
    in most cases equal to 2 (and by default in out case)

    Were presented in first paper about ResNet (i.e. v1)

    Parameters
    ----------
    x : mf.MakiTensor
        Input mf.MakiTensor.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    stride : int
        Stride for this block, by defualt equal to 2
    out_f : int
        Output number of feature maps.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    reduction : int
        The number of feature maps to which you want to increase/decrease,
        By default decrease input feature maps by 2
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : mf.MakiTensor
        Output mf.MakiTensor.

    """
    if bn_params is None:
        bn_params = get_batchnorm_params()

    prefix_name = PREFIX_NAME_BLOCK.format(block_id, unit_id)

    if in_f is None:
        in_f = x.shape()[-1]
    if reduction is None:
        reduction = int(in_f / 2)
    if out_f is None:
        out_f = in_f * 2

    # Main Branch
    # Conv(1x1) -> BN -> Activ
    mx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=reduction, stride=stride, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 1, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(x)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 1, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 1, ACTIV))(mx)

    # Conv(3x3) -> BN -> Activ
    mx = ConvLayer(
        kw=3, kh=3, in_f=reduction, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 2, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(mx)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 2, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 2, ACTIV))(mx)

    # Conv(1x1) -> BN
    mx = ConvLayer(
        kw=1, kh=1, in_f=reduction, out_f=out_f, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 3, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(mx)
    mx = BatchNormLayer(D=out_f, name=PREFIX_NAME_LAYER.format(prefix_name, 3, BATCH_NORM), **bn_params)(mx)

    # Skip branch
    sx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=out_f, stride=stride, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_SHORTCUT.format(prefix_name, WEIGHTS),
        kernel_initializer=kernel_initializer
    )(x)
    sx = BatchNormLayer(D=out_f, name=PREFIX_NAME_SHORTCUT.format(prefix_name, BATCH_NORM), **bn_params)(sx)

    x = SumLayer(name=prefix_name + str(num_block) + SUM_OPERATION)([mx,sx])

    return x


def ResNetIdentityBlock_woPointWiseV1(
        x : mf.MakiTensor,
        block_id: int,
        unit_id: int,
        num_block=None,
        in_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        kernel_initializer=HE_INIT,
        bn_params=None):
    """
    Create ResNet block with skip connection and without pointwise operation in block,
    This type of blocks in most cases are used in ResNet34, ResNet18

    Were presented in first paper about ResNet (i.e. v1)

    Parameters
    ----------
    x : mf.MakiTensor
        Input mf.MakiTensor.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : mf.MakiTensor
        Output mf.MakiTensor.

    """
    if bn_params is None:
        bn_params = get_batchnorm_params()

    prefix_name = PREFIX_NAME_BLOCK_WOP.format(block_id, unit_id)

    if in_f is None:
        in_f = x.shape()[-1]

    # BN -> ACT -> ZERO_PADDING -> CONV, first block
    mx = BatchNormLayer(D=in_f, name=BN.format(prefix_name, 1), **bn_params)(x)
    mx = ActivationLayer(activation=activation, name=ACTIVATION.format(prefix_name, 1))(mx)
    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=ZERO_PADDING.format(prefix_name, 1))(mx)
    mx = ConvLayer(
        kw=3, kh=3, in_f=in_f, out_f=in_f, activation=None,
        padding='VALID', use_bias=use_bias, name=CONV.format(prefix_name, 1),
        kernel_initializer=kernel_initializer
    )(mx)

    # BN -> ACT -> ZERO_PADDING -> CONV, second block
    mx = BatchNormLayer(D=in_f, name=BN.format(prefix_name, 2), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=ACTIVATION.format(prefix_name, 2))(mx)
    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=ZERO_PADDING.format(prefix_name, 2))(mx)
    mx = ConvLayer(
        kw=3, kh=3, in_f=in_f, out_f=in_f, activation=None,
        padding='VALID', use_bias=use_bias, name=CONV.format(prefix_name, 2),
        kernel_initializer=kernel_initializer
    )(mx)

    x = SumLayer(name=prefix_name + SUM_OPERATION + str(num_block))([mx,x])

    return x


def ResNetConvBlock_woPointWiseV1(
        x : mf.MakiTensor,
        block_id: int,
        unit_id: int,
        num_block: int,
        activation=tf.nn.relu,
        use_bias=False,
        stride=2,
        in_f=None,
        out_f=None,
        kernel_initializer=InitConvKernel.HE,
        bn_params=None):
    """
    Create ResNet block with skip connection using certain `stride`
    And without point wise convolutions (i.e. convs with 1x1 kernel),
    in most cases `stride` equal to 2 (and by default in out case)

    Were presented in first paper about ResNet (i.e. v1)

    Parameters
    ----------
    x : mf.MakiTensor
        Input mf.MakiTensor.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    activation : tensorflow function
        The function of activation. By default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    stride : int
        Stride for this block, by defualt equal to 2
    in_f : int
        Number of input feature maps. By default is None (shape will be getted from tensor).
    out_f : int
        Number of output feature maps. By default is None which means out_f = 2 * in_f.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : mf.MakiTensor
        Output mf.MakiTensor.

    """
    if bn_params is None:
        bn_params = get_batchnorm_params()

    prefix_name = PREFIX_NAME_BLOCK_WOP.format(block_id, unit_id)

    if in_f is None:
        in_f = x.shape()[-1]
    if out_f is None:
        out_f = int(2*in_f)

    # BatchNorm + activation layer before main ConvBlock
    x = BatchNormLayer(D=in_f, name=BN.format(prefix_name, 1), **bn_params)(x)
    x = ActivationLayer(activation=activation, name=ACTIVATION.format(prefix_name, 1))(x)

    # Main branch
    # Zero_padding -> Conv -> BN -> Activation -> Zero_padding -> Conv
    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=ZERO_PADDING.format(prefix_name, 1))(x)
    mx = ConvLayer(
        kw=3, kh=3, in_f=in_f, out_f=out_f, activation=None, stride=stride,
        padding='VALID', use_bias=use_bias, name=CONV.format(prefix_name, 1),
        kernel_initializer=kernel_initializer
    )(mx)

    mx = BatchNormLayer(D=out_f, name=BN.format(prefix_name, 2), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=ACTIVATION.format(prefix_name, 2))(mx)
    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=ZERO_PADDING.format(prefix_name, 2))(mx)
    mx = ConvLayer(
        kw=3, kh=3, in_f=out_f, out_f=out_f, activation=None,
        padding='VALID', use_bias=use_bias, name=CONV.format(prefix_name, 2),
        kernel_initializer=kernel_initializer
    )(mx)
                                                                                
    # Skip branch
    sx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=out_f, stride=stride,
        padding='VALID', activation=None, use_bias=use_bias, name=SCIP_BRANCH.format(prefix_name),
        kernel_initializer=kernel_initializer
    )(x)
                                                                               
    x = SumLayer(name=prefix_name + SUM_OPERATION + str(num_block))([mx, sx])

    return x
