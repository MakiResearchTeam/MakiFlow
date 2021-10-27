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
from .builder import build_DenseNet


def DenseNet121(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create DenseNet121 model with certain input `in_x`

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
    return build_DenseNet(
        in_x=in_x,
        nb_layers=[6, 12, 24, 16],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet121',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bn_params={}
    )


def DenseNet161(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create DenseNet161 model with certain input `in_x`

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
    return build_DenseNet(
        in_x=in_x,
        nb_layers=[6, 12, 36, 24],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet161',
        growth_rate=24,
        reduction=0.5,
        dropout_p_keep=0.8,
        bn_params={}
    )


def DenseNet169(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create DenseNet169 model with certain input `in_x`

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
    return build_DenseNet(
        in_x=in_x,
        nb_layers=[6, 12, 32, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet169',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bn_params={}
    )


def DenseNet201(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create DenseNet201 model with certain input `in_x`

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
    return build_DenseNet(
        in_x=in_x,
        nb_layers=[6, 12, 48, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet201',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bn_params={}
    )


def DenseNet264(
        in_x: mf.MakiTensor, classes=1000,
        include_top=False, create_model=False):
    """
    Create DenseNet264 model with certain input `in_x`

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
    return build_DenseNet(
        in_x=in_x,
        nb_layers=[6, 12, 64, 48],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet264',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bn_params={}
    )

