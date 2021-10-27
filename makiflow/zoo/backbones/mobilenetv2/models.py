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

import makiflow as mf
import tensorflow as tf

from .builder import build_MobileNetV2


def MobileNetV2_1_0(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create MobileNetV2 with alpha=1.0 model with certain `input_shape`

    Parameters
    ----------
    in_x : mf.MakiTensor
        A tensor that will be fed into the model as input tensor.
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj

    Returns
    -------
    output : mf.MakiTensor
        Output MakiTensor. if `create_model` is False
    model : mf.Model
        MakiFlow model. if `create_model` is True

    """
    return  build_MobileNetV2(
        in_x=in_x,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_0',
        alpha=1.0,
        expansion=6
    )


def MobileNetV2_1_4(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create MobileNetV2 with alpha=1.4  model with certain `input_shape`

    Parameters
    ----------
    in_x : mf.MakiTensor
        A tensor that will be fed into the model as input tensor.
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj

    Returns
    -------
    output : mf.MakiTensor
        Output MakiTensor. if `create_model` is False
    model : mf.Model
        MakiFlow model. if `create_model` is True

    """
    return  build_MobileNetV2(
        in_x=in_x,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_4',
        alpha=1.4,
        expansion=6
    )


def MobileNetV2_0_75(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create MobileNetV2 with alpha=0.75  model with certain `input_shape`

    Parameters
    ----------
    in_x : mf.MakiTensor
        A tensor that will be fed into the model as input tensor.
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj

    Returns
    -------
    output : mf.MakiTensor
        Output MakiTensor. if `create_model` is False
    model : mf.Model
        MakiFlow model. if `create_model` is True

    """
    return  build_MobileNetV2(
        in_x=in_x,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_0_75',
        alpha=0.75,
        expansion=6
    )


def MobileNetV2_1_3(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create MobileNetV2 with alpha=1.3  model with certain `input_shape`

    Parameters
    ----------
    in_x : mf.MakiTensor
        A tensor that will be fed into the model as input tensor.
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj

    Returns
    -------
    output : mf.MakiTensor
        Output MakiTensor. if `create_model` is False
    model : mf.Model
        MakiFlow model. if `create_model` is True

    """
    return build_MobileNetV2(
        in_x=in_x,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_3',
        alpha=1.3,
        expansion=6
    )
