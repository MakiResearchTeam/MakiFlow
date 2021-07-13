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
from .blocks import transition_layer, dense_block
from .utils import get_batchnorm_params

from makiflow.layers import *
from makiflow import Model


def build_DenseNet(
        in_x,
        nb_layers=[6, 12, 24, 16],
        depth=121,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=False,
        name_model='MakiClassificator',
        growth_rate=32,
        reduction=0.0,
        nb_blocks=3,
        dropout_p_keep=None,
        bn_params={}
):
    """
    Parameters
    ----------
    in_x : MakiTensor
        Input MakiTensor.
    nb_layers : int
        List of length 4, where `nb_layers[i]` is number of repetition layers at stage `i` (i from 0 to 3).
    depth : int
        If list `nb_layers` will be empty, number of blocks will be calculated based on `depth`
        For example: if depth=40, then first 4 blocks will repeat repeat_times = int((depth - 4) / 3), if use_bottleneck is True when
        repeat_times divide by 2, in these example repeat_times = 6, so `nb_layers` = [6,6,6,6] in these example.
    growth_rate : int
        Coefficient `k` from original paper, https://arxiv.org/pdf/1608.06993.pdf .
    dropout_p_keep : float
        The probability that each element of x is not discarded.
    reduction : float
        Coefficient, where `r` = 1 - `rediction`, `r` is how much number of feature maps need to compress in transition layers.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    nb_blocks : int
        Number of dense blocks.
    num_classes : int
        Number of classes that you need to classify.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.

     Returns
     ---------
     in_x : MakiTensor
         Input MakiTensor.
     output : int
         Output MakiTensor.
     model : MakiFlow.Model
         Constructed model.
     """
    if bn_params is None or len(bn_params) == 0:
        bn_params = get_batchnorm_params()
    compression = 1 - reduction

    if len(nb_layers) == 0:
        count = int((depth - 4) / 3)
        if use_bottleneck:
            count //= 2
        nb_layers = [count for _ in range(nb_blocks + 1)]

    if subsample_initial_block:
        x = ZeroPaddingLayer(padding=[[3, 3], [3, 3]], name='zero_padding2d_4')(in_x)

        x = ConvLayer(kw=7, kh=7, in_f=3, stride=2, out_f=growth_rate * 2, activation=None, use_bias=use_bias,
                      name='conv1/conv', padding='VALID')(x)

        x = BatchNormLayer(D=growth_rate * 2, name='conv1/bn', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1/relu')(x)
        x = ZeroPaddingLayer(padding=[[1, 1], [1, 1]], name='zero_padding2d_5')(x)

        x = MaxPoolLayer(ksize=[1, 3, 3, 1], padding='VALID', name='pool1')(x)
    else:
        x = ConvLayer(kw=3, kh=3, in_f=3, stride=1, out_f=growth_rate * 2, activation=None, use_bias=use_bias,
                      name='conv1/conv')(in_x)

    # densenet blocks
    for block_index in range(len(nb_layers) - 1):
        # dense block
        x = dense_block(x=x, nb_layers=nb_layers[block_index], stage=block_index + 2,
                        growth_rate=growth_rate, dropout_p_keep=dropout_p_keep, use_bottleneck=use_bottleneck,
                        activation=activation, use_bias=use_bias, bn_params=bn_params)

        # transition block
        x = transition_layer(x=x,
                             dropout_p_keep=dropout_p_keep, number=block_index + 2, compression=compression,
                             activation=activation, use_bias=use_bias, bn_params=bn_params)

    x = dense_block(x=x, nb_layers=nb_layers[-1], stage=len(nb_layers) + 1,
                    growth_rate=growth_rate, dropout_p_keep=dropout_p_keep, use_bottleneck=use_bottleneck,
                    activation=activation, use_bias=use_bias, bn_params=bn_params)

    x = BatchNormLayer(D=x.shape[-1], name='bn', **bn_params)(x)
    x = ActivationLayer(activation=activation, name='relu')(x)
    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        # dense part (fc layers)
        output = DenseLayer(in_d=x.shape[-1], out_d=num_classes, activation=None, use_bias=True, name="fc1000")(x)
    else:
        output = x

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)

    return output
