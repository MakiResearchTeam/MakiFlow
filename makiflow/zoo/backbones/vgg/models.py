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
from .builder import build_VGG


def VGG16(in_x, classes=1000, include_top=False, create_model=False):
    return build_VGG(
        in_x=in_x,
        repetition=3,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='VGG16'
    )


def VGG19(in_x, classes=1000, include_top=False, create_model=False):
    return build_VGG(
        in_x=in_x,
        repetition=4,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='VGG19'
    )
