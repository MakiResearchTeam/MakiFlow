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

from makiflow.core import MakiTensor
import tensorflow as tf

from .utils import get_pool_params


PREFIX = "conv{}/conv{}_"

NAME_CONV = "{}{}"
NAME_ACT = "{}activation_{}"
NAME_POOL = "block{}_pool"

MAX_POOL = 'max_pool'
AVG_POOL = 'avg_pool'
NONE = 'none'


def VGGBlock(
        x: MakiTensor,
        num_block: str,
        n: int,
        in_f=None,
        out_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        pooling_type='max_pool',
        kernel_initializer=InitConvKernel.HE,
        pool_params=None):
    """
    VGGBlock is consist of certain number (in our case `n`) conv + activation layers,
    First layer scale `in_f` upto `out_f`


    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    num_block : str
        Number of block (used in name of layers).
    n : int
        Number of convolutions in blocks.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    out_f : int
        Number of output feature maps. By default None (shape will same as `in_f` * 2).
    use_bias : bool
        Use bias on layers or not.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    pooling_type : str
        What type of pooling are will be used.
        'max_pool' - for max pooling.
        'avg_pool' - for average pooling.
        'none' or any other strings - the operation pooling will not be used.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    pool_params : dict
        Parameters for pool layer. If equal to None then all parameters will have default valued.
        Default parameters are:
        {
            'ksize': [1,2,2,1],
            'strides': [1,2,2,1],
            'padding': 'SAME'
        }

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.

    """

    if pool_params is None:
        pool_params = get_pool_params()

    prefix_name = PREFIX.format(str(num_block), str(num_block))

    if in_f is None:
        in_f = x.get_shape()[-1]

    if out_f is None:
        out_f = in_f * 2

    x = ConvLayer(
        kw=3,kh=3,in_f=in_f,out_f=out_f,use_bias=use_bias,
        activation=None,name=NAME_CONV.format(prefix_name, str(1)),
        kernel_initializer=kernel_initializer
    )(x)
    x = ActivationLayer(activation=activation, name=NAME_ACT.format(prefix_name, str(1)))(x)

    for i in range(2, n+1):
        x = ConvLayer(
            kw=3,kh=3,in_f=out_f,out_f=out_f,use_bias=use_bias,
            activation=None,name=NAME_CONV.format(prefix_name, str(i)),
            kernel_initializer=kernel_initializer
        )(x)
        x = ActivationLayer(activation=activation, name=NAME_ACT.format(prefix_name, str(i)))(x)

    if pooling_type == MAX_POOL:
        x = MaxPoolLayer(
            name=NAME_POOL.format(str(num_block)), **pool_params
        )(x)
    elif pooling_type == AVG_POOL:
        x = AvgPoolLayer(
            name=NAME_POOL.format(str(num_block)), **pool_params
        )(x)

    return x

