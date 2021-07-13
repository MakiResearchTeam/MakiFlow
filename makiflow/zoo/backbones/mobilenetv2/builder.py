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
from .blocks import inverted_res_block
from .utils import make_divisible, get_batchnorm_params

from makiflow.layers import *
from makiflow import Model, MakiTensor


def build_MobileNetV2(
        in_x: MakiTensor,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=False,
        name_model='MakiClassificator',
        alpha=1,
        expansion=6,
        stride_list=(2, 2, 2, 2, 2),
        bn_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    expansion : int
        Magnification multiplier of feature maps.
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    alpha : int
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, default number of filters from the paper are used at each layer.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
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
    create_model : bool
        Return build classification model, otherwise return input mf.MakiTensor and output mf.MakiTensor.
    name_model : str
        Name of model, if it will be created.
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.
    num_classes : int
        Number of classes that you need to classify.

    Returns
    ---------
    in_x : mf.MakiTensor
        Input mf.MakiTensor.
    output : int
        Output mf.MakiTensor.
    Classificator : mf.models.Classificator
        Constructed model.
    """
    if bn_params is None or len(bn_params) == 0:
        bn_params = get_batchnorm_params()

    first_filt = make_divisible(32 * alpha, 8)

    x = ConvLayer(kw=3,
                  kh=3,
                  in_f=in_x.shape[-1],
                  out_f=first_filt,
                  stride=stride_list[0],
                  padding='SAME',
                  activation=None,
                  use_bias=use_bias,
                  name='Conv/weights',
                  )(in_x)

    x = BatchNormLayer(D=first_filt, name='Conv/BatchNorm', **bn_params)(x)
    x = ActivationLayer(activation=activation, name='Conv_relu')(x)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=16, alpha=alpha, expansion=1,
                           block_id=0, use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=False, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=24, alpha=alpha, stride=stride_list[1],
                           expansion=expansion,
                           block_id=1, use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=24, alpha=alpha, expansion=expansion, block_id=2,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=32, alpha=alpha, stride=stride_list[2],
                           expansion=expansion,
                           block_id=3, use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=32, alpha=alpha, expansion=expansion, block_id=4,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=32, alpha=alpha, expansion=expansion, block_id=5,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=64, alpha=alpha, stride=stride_list[3],
                           expansion=expansion,
                           block_id=6, use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=64, alpha=alpha, expansion=expansion, block_id=7,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=64, alpha=alpha, expansion=expansion, block_id=8,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=64, alpha=alpha, expansion=expansion, block_id=9,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=96, alpha=alpha, expansion=expansion, block_id=10,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=96, alpha=alpha, expansion=expansion, block_id=11,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=96, alpha=alpha, expansion=expansion, block_id=12,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=160, alpha=alpha, stride=stride_list[4],
                           expansion=expansion, block_id=13,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=160, alpha=alpha, expansion=expansion, block_id=14,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=160, alpha=alpha, expansion=expansion, block_id=15,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(x=x, in_f=x.shape[-1], out_f=320, alpha=alpha, expansion=expansion, block_id=16,
                           use_bias=use_bias, activation=activation,
                           bn_params=bn_params, use_expand=True, use_skip_connection=False)

    if alpha > 1.0:
        last_block_filters = make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = ConvLayer(kh=1,
                  kw=1,
                  in_f=x.shape[-1],
                  out_f=last_block_filters,
                  stride=1,
                  padding='SAME',
                  activation=None,
                  use_bias=use_bias,
                  name='Conv_1/weights',
                  )(x)

    x = BatchNormLayer(D=last_block_filters, name='Conv_1/BatchNorm', **bn_params)(x)
    pred_top = ActivationLayer(activation=activation, name='out_relu')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='global_avg')(pred_top)
        x = ReshapeLayer(new_shape=[1, 1, 1280], name='resh')(x)
        x = ConvLayer(kw=1, kh=1, in_f=1280, out_f=num_classes, name='prediction')(x)
        output = ReshapeLayer(new_shape=[num_classes], name='endo')(x)
    else:
        output = pred_top

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)

    return in_x, output
