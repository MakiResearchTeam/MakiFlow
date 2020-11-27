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

from makiflow.layers import ConvLayer, InputLayer
from .ssp_model import SSPModel
from .head import Head
from .embedding_layer import SkeletonEmbeddingLayer

N_POINTS = 15


def make_head(x, ind):
    _, _, _, c = x.get_shape()
    offsets = ConvLayer(
        kw=1, kh=1, in_f=c, out_f=N_POINTS*2, activation=None, name=f'offsets_{ind}'
    )(x)
    coords = SkeletonEmbeddingLayer(embedding_dim=N_POINTS, name=f'embedding_{ind}')(offsets)
    point_indicators = ConvLayer(
        kw=1, kh=1, in_f=c, out_f=N_POINTS, activation=None, name=f'point_indicators_{ind}'
    )(x)
    human_indicators = ConvLayer(
        kw=1, kh=1, in_f=c, out_f=1, activation=None, name=f'human_indicators_{ind}'
    )(x)
    return Head(
        coords=coords,
        point_indicators=point_indicators,
        human_indicators=human_indicators
    )


def ssp_model():
    in_x = InputLayer([1, 64, 64, 3], name='input_image')
    x = ConvLayer(kw=3, kh=3, in_f=3, out_f=64, stride=2, name='conv1_1')(in_x)
    x = ConvLayer(kw=3, kh=3, in_f=64, out_f=64, stride=2, name='conv1_2')(x)   # [16, 16]
    head1 = make_head(x, 1)
    x = ConvLayer(kw=3, kh=3, in_f=64, out_f=64, stride=2, name='conv2')(x)     # [8, 8]
    head2 = make_head(x, 2)
    x = ConvLayer(kw=3, kh=3, in_f=64, out_f=64, stride=2, name='conv3')(x)     # [4, 4]
    head3 = make_head(x, 3)
    x = ConvLayer(kw=3, kh=3, in_f=64, out_f=64, stride=2, name='conv4')(x)     # [2, 2]
    head4 = make_head(x, 4)
    x = ConvLayer(kw=1, kh=1, in_f=64, out_f=64, stride=2, name='conv5')(x)     # [1, 1]
    head5 = make_head(x, 5)

    model = SSPModel(
        in_x=in_x,
        heads=[head1, head2, head3, head4, head5]
    )
    return model
