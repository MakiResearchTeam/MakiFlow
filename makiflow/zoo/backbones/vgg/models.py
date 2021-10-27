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
from makiflow.layers.utils import InitConvKernel
from .builder import build_VGG

# Preprocess:
#         B        G        R
# Image - [103.939, 116.779, 123.68]


def VGG16(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create VGG16 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full VGG16 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """

    return build_VGG(
            input_shape=input_shape,
            repetition=3,
            include_top=include_top,
            num_classes=classes,
            use_bias=True,
            activation=tf.nn.relu,
            create_model=create_model,
            kernel_initializer=kernel_initializer,
            name_model='VGG16'
    )


def VGG19(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create VGG19 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full VGG19 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """

    return build_VGG(
            input_shape=input_shape,
            repetition=4,
            include_top=include_top,
            num_classes=classes,
            use_bias=True,
            activation=tf.nn.relu,
            create_model=create_model,
            kernel_initializer=kernel_initializer,
            name_model='VGG19'
    )

