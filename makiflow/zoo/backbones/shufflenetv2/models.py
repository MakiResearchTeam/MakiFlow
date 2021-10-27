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
from .builder import build_ShuffleNetV2
from .utils import MODEL_20, MODEL_15, MODEL_10, MODEL_05, SIZE_2_CONFIG_MODEL


def ShuffleNetV2_20(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ShuffleNetV2 with x2.0 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ShuffleNetV2 with x2.0 model
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
    return build_ShuffleNetV2(
        input_shape=input_shape,
        model_config=SIZE_2_CONFIG_MODEL[MODEL_20],
        shuffle_group=2,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ShuffleNetv2_20'
    )


def ShuffleNetv2_15(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ShuffleNetV2 with x1.5 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ShuffleNetV2 with x1.5 model
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
    return build_ShuffleNetV2(
        input_shape=input_shape,
        model_config=SIZE_2_CONFIG_MODEL[MODEL_15],
        shuffle_group=2,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ShuffleNetv2_20'
    )


def ShuffleNetv2_10(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ShuffleNetV2 with x1.0 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ShuffleNetV2 with x1.0 model
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
    return build_ShuffleNetV2(
        input_shape=input_shape,
        model_config=SIZE_2_CONFIG_MODEL[MODEL_10],
        shuffle_group=2,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ShuffleNetv2_20'
    )


def ShuffleNetv2_05(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ShuffleNetV2 with x0.5 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ShuffleNetV2 with x0.5 model
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
    return build_ShuffleNetV2(
        input_shape=input_shape,
        model_config=SIZE_2_CONFIG_MODEL[MODEL_05],
        shuffle_group=2,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ShuffleNetv2_20'
    )
