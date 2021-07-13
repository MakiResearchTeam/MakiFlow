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
from makiflow import Model


from .utils import get_batchnorm_params, get_pool_params
from .blocks import repeat_n_convLayers


def build_VGG(
    in_x,
    repetition=3,
    number_of_blocks=5,
    include_top=False,
    num_classes=1000,
    use_bias=False,
    activation=tf.nn.relu,
    create_model=False,
    init_fm=64,
    input_tensor=None,
    pooling_type='max_pool',
    name_model='MakiClassificator'):
    """
    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    repetition : int
        Number of repetition convolution per block, usually 3 for VGG16, 4 for vgg 19.
    number_of_blocks : int
        Number of blocks of `repetition`.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
    use_bias : bool
        If true, when on layers used bias operation.
    init_fm : int
        Initial number of feature maps.
    activation : tf object
        Activation on every convolution layer.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    pooling_type : str
        What type of pooling are will be used.
        'max_pool' - for max pooling.
        'avg_pool' - for average pooling.
        'none' or any other strings - the operation pooling will not be used.
    name_model : str
        Name of model, if it will be created.
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.

    Returns
    ---------
    in_x : mf.MakiTensor
        Input MakiTensor.
    output : mf.MakiTensor
        Output MakiTensor.
    Classificator : mf.Model
        Constructed model.
    """

    if repetition <= 0:
        raise TypeError('repetition should have type int and be more than 0')

    bn_params = get_batchnorm_params()
    pool_params = get_pool_params()

    for i in range(1, number_of_blocks):
        # First block
        if i == 1:
            x = repeat_n_convLayers(in_x, num_block=i, n=2, out_f=init_fm, pooling_type=pooling_type,
                                    bn_params=bn_params, pool_params=pool_params)
        # Second block
        elif i == 2:
            x = repeat_n_convLayers(x, num_block=i, n=2, pooling_type=pooling_type,
                                    bn_params=bn_params, pool_params=pool_params)
        else:
            x = repeat_n_convLayers(x, num_block=i, n=repetition, pooling_type=pooling_type,
                                    bn_params=bn_params, pool_params=pool_params)

    # Last block
    x = repeat_n_convLayers(x, out_f=x.shape[-1], num_block=number_of_blocks,
                            n=repetition, pooling_type=pooling_type,
                            bn_params=bn_params, pool_params=pool_params)

    if include_top:
        x = FlattenLayer(name='flatten')(x)
        in_f = x.shape[1] * x.shape[2] * x.shape[3]
        x = DenseLayer(in_d=in_f, out_d=4096, name='fc6')(x)
        x = DenseLayer(in_d=4096, out_d=4096, name='fc7')(x)
        output = DenseLayer(in_d=4096, out_d=num_classes, activation=None, name='fc8')(x)
    else:
        output = x

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)

    return in_x, output
