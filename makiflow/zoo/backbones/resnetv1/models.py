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


from .builder import build_ResNetV1, build_LittleResNetV1
from .utils import get_head_batchnorm_params, WITHOUT_POINTWISE, WITH_POINTWISE
from makiflow.layers.utils import InitConvKernel
import tensorflow as tf


# --------------------------------------------------------------------------------
#   Standard Residual Models V1
# --------------------------------------------------------------------------------

def ResNet18(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
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
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(2, 2, 2, 2),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=False,
        use_bias=False,
        activation=tf.nn.relu,
        block_type=WITHOUT_POINTWISE,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ResNet18'
    )


def ResNet34(
        input_shape,
        classes=1000,
        include_top=False,
        factorization_first_layer=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet34 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet34 model
    factorization_first_layer : bool
        If true at the start of CNN factorize convolution layer into 3 convolution layers.
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
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 6, 3),
        include_top=include_top,
        num_classes=classes,
        using_zero_padding=True,
        head_bn_params=get_head_batchnorm_params(),
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type=WITHOUT_POINTWISE,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ResNet34'
    )


def ResNet50(
        input_shape,
        classes=1000,
        include_top=False,
        factorization_first_layer=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet50 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet50 model
    factorization_first_layer : bool
        If true at the start of CNN factorize convolution layer into 3 convolution layers.
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
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 6, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type=WITH_POINTWISE,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ResNet50',
    )


def ResNet101(
        input_shape,
        classes=1000,
        include_top=False,
        factorization_first_layer=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet101 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet101 model
    factorization_first_layer : bool
        If true at the start of CNN factorize convolution layer into 3 convolution layers.
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
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 23, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type=WITH_POINTWISE,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ResNet101'
    )


def ResNet152(
        input_shape,
        classes=1000,
        include_top=False,
        factorization_first_layer=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet152 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet152 model
    factorization_first_layer : bool
        If true at the start of CNN factorize convolution layer into 3 convolution layers.
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

    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 8, 36, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type=WITH_POINTWISE,
        create_model=create_model,
        kernel_initializer=kernel_initializer,
        name_model='ResNet152'
    )


# --------------------------------------------------------------------------------
#   Little version of the Residual Models V1
# --------------------------------------------------------------------------------
# Implementation taken from https://keras.io/examples/cifar10_resnet/


def Little_ResNet20(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create little ResNet20 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full little ResNet20 model
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

    return build_LittleResNetV1(
        input_shape,
        depth=20,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet20',
        kernel_initializer=kernel_initializer,
        activation_between_blocks=True
    )


def Little_ResNet32(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create little ResNet32 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full little ResNet32 model
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

    return build_LittleResNetV1(
        input_shape,
        depth=32,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet32',
        kernel_initializer=kernel_initializer,
        activation_between_blocks=True
    )


def Little_ResNet44(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create little ResNet44 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full little ResNet44 model
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

    return build_LittleResNetV1(
        input_shape,
        depth=44,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet44',
        kernel_initializer=kernel_initializer,
        activation_between_blocks=True
    )


def Little_ResNet56(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create little ResNet56 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full little ResNet56 model
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

    return build_LittleResNetV1(
        input_shape,
        depth=56,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet56',
        kernel_initializer=kernel_initializer,
        activation_between_blocks=True
    )


def Little_ResNet110(
        input_shape,
        classes=1000,
        include_top=False,
        create_model=False,
        kernel_initializer=InitConvKernel.HE):
    """
    Create little ResNet110 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full little ResNet110 model
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

    return build_LittleResNetV1(
        input_shape,
        depth=110,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet110',
        kernel_initializer=kernel_initializer,
        activation_between_blocks=True
    )

