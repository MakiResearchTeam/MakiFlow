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

from makiflow.layers import *


def transition_layer(
        x,
        dropout_p_keep,
        number,
        compression=1.0,
        activation=tf.nn.relu,
        use_bias=False,
        bn_params={}
):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    dropout_p_keep : int
        The probability that each element of x is not discarded.
    number : int
        Number of transition block (used in name of layers).
    compression : float
        How much number of feature maps need to compress.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """
    assert 0.0 <= compression <= 1.0, f"wrong value for compression: {compression}"

    prefix = f'pool{str(number)}_'

    in_f = x.get_shape()[-1]
    out_f = int(in_f * compression)

    x = BatchNormLayer(D=in_f, name=prefix + 'bn', **bn_params)(x)
    x = ActivationLayer(activation=activation, name=prefix + 'relu')(x)
    x = ConvLayer(kw=1, kh=1, in_f=in_f, out_f=out_f, activation=None, use_bias=use_bias,
                  name=prefix + 'conv', padding='VALID')(x)
    if dropout_p_keep is not None:
        x = DropoutLayer(p_keep=dropout_p_keep, name=prefix + 'dropout')(x)
    x = AvgPoolLayer(padding='VALID', name=prefix + 'avg_pool')(x)

    return x


def conv_layer(
        x,
        growth_rate,
        dropout_p_keep,
        stage,
        block,
        multiply=4,
        use_bottleneck=True,
        activation=tf.nn.relu,
        use_bias=False,
        bn_params={}
):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    growth_rate : int
        Coefficient `k` from original papep, https://arxiv.org/pdf/1608.06993.pdf .
    dropout_p_keep : int
        The probability that each element of x is not discarded.
    use_bottleneck : bool
        Use bottleneck block or not.
    stage : int
        Number of stage (used in name of layers).
    block : int
        Number of bottleneck block (used in name of layers).
    multiply : int
        The multiplier of the `growth_rate`.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """
    prefix = f'conv{str(stage)}_block{str(block)}_'

    in_f = x.get_shape()[-1]

    x = BatchNormLayer(D=in_f, name=prefix + '0_bn', **bn_params)(x)
    x = ActivationLayer(activation=activation, name=prefix + '0_relu')(x)
    if use_bottleneck:
        growth_f = multiply * growth_rate
        x = ConvLayer(kw=1, kh=1, in_f=in_f, out_f=growth_f, activation=None, use_bias=use_bias,
                      name=prefix + '1_conv', padding='VALID')(x)

        if dropout_p_keep is not None:
            x = DropoutLayer(p_keep=dropout_p_keep, name=prefix + '1_dropout')(x)

        x = BatchNormLayer(D=growth_f, name=prefix + '1_bn', **bn_params)(x)
        x = ActivationLayer(activation=activation, name=prefix + '1_relu')(x)

    x = ConvLayer(kw=3, kh=3, in_f=x.get_shape()[-1], out_f=growth_rate, activation=None, use_bias=use_bias,
                  name=prefix + '2_conv')(x)

    if dropout_p_keep is not None:
        x = DropoutLayer(p_keep=dropout_p_keep, name=prefix + '2_dropout')(x)

    return x


def dense_block(
        x,
        nb_layers,
        stage,
        growth_rate,
        dropout_p_keep,
        use_bottleneck=True,
        activation=tf.nn.relu,
        use_bias=False,
        bn_params={}
):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    growth_rate : int
        Coefficient `k` from original papep, https://arxiv.org/pdf/1608.06993.pdf .
    dropout_p_keep : int
        The probability that each element of x is not discarded.
    use_bottleneck : bool
        Use bottleneck block or not in conv_layer.
    stage : int
        Number of stage (used in name of layers).
    nb_layers : int
        Number of layers in DenseBlock.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    concat_layers : MakiTensor
        Output MakiTensor.
    """
    concat_layers = x

    for i in range(nb_layers):
        x = conv_layer(x=concat_layers, growth_rate=growth_rate, dropout_p_keep=dropout_p_keep, stage=stage,
                       block=i + 1,
                       activation=activation, use_bottleneck=use_bottleneck, use_bias=use_bias, bn_params=bn_params)

        concat_layers = ConcatLayer(name=f'conv{stage}_block{i + 1}_concat')([concat_layers, x])

    return concat_layers
