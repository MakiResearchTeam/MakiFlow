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


from .blocks import (ResNetIdentityBlockV1, ResNetConvBlockV1,
                     ResNetIdentityBlock_woPointWiseV1, ResNetConvBlock_woPointWiseV1)


from .utils import (get_batchnorm_params, get_head_batchnorm_params,
                    get_batchnorm_params_resnet34, WITH_POINTWISE, WITHOUT_POINTWISE)

from makiflow.layers import *
from makiflow.layers.utils import InitConvKernel
from makiflow.models import Classificator
import tensorflow as tf


def build_ResNetV1(
    input_shape=None,
    repetition=(2,2,2,2),
    include_top=False,
    num_classes=1000,
    factorization_first_layer=False,
    use_bias=False,
    using_zero_padding=False,
    stride_list=(2, 2, 2, 2, 2),
    head_bn_params=None,
    activation=tf.nn.relu,
    block_type=WITH_POINTWISE,
    create_model=False,
    name_model='MakiClassificator',
    init_filters=64,
    min_reduction=64,
    activation_between_blocks=True,
    kernel_initializer=InitConvKernel.HE,
    output_factorization_layer=None,
    input_tensor=None):
    """
    Build ResNet version 1 with certain parameters

    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    repetition : list
        Number of repetition on certain depth.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
    factorization_first_layer : bool
        If true at the start of CNN factorize convolution layer into 3 convolution layers.
    use_bias : bool
        If true, when on layers used bias operation.
    activation : tf object
        Activation on every convolution layer.
    block_type : str
        Type of blocks.
        with_pointwise - use pointwise operation in blocks, usually used in ResNet50, ResNet101, ResNet152.
        without_pointwise - block without pointwise operation, usually  used in ResNet18, ResNet34.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    init_filters : int
        Started number of feature maps.
    min_reduction : int
        Minimum reduction in blocks.
    activation_between_blocks : bool
        Use activation between blocks.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    output_factorization_layer : int
        Number of output featues from factorized layer, if equal to None,
        If use pointwise output_factorization_layer = init_filters
        otherwise output_factorization_layer = 2 * output_factorization_layer
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.

    Returns
    ---------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """

    if (type(repetition) is not list and type(repetition) is not tuple) or len(repetition) != 4:
        raise TypeError('repetition should be list of size 4')

    feature_maps = init_filters
    if using_zero_padding:
        bn_params = get_batchnorm_params_resnet34()
    else:
        bn_params = get_batchnorm_params()

    if block_type == WITH_POINTWISE:
        conv_block = ResNetConvBlockV1
        iden_block = ResNetIdentityBlockV1
        if output_factorization_layer is None:
            output_factorization_layer = init_filters
        pointwise = True
    elif block_type == WITHOUT_POINTWISE:
        conv_block = ResNetConvBlock_woPointWiseV1
        iden_block = ResNetIdentityBlock_woPointWiseV1
        if output_factorization_layer is None:
            output_factorization_layer = init_filters * 2
        pointwise = False
    else:
        raise Exception(f'{block_type} type is not found')

    if input_tensor is None:
        in_x = InputLayer(input_shape=input_shape, name='Input')
    elif input_tensor is not None:
        in_x = input_tensor
        input_shape = input_tensor.get_shape()

    if factorization_first_layer:

        x = ConvLayer(
            kw=3, kh=3, in_f=input_shape[-1], out_f=feature_maps, use_bias=use_bias,
            activation=None, stride=stride_list[0], name='conv1_1/weights',
            kernel_initializer=kernel_initializer
        )(in_x)
        x = BatchNormLayer(D=feature_maps, name='conv1_1/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_1/activation')(x)

        x = ConvLayer(
            kw=3, kh=3, in_f=feature_maps, out_f=feature_maps, use_bias=use_bias,
            activation=None, name='conv1_2/weights',
            kernel_initializer=kernel_initializer
        )(x)
        x = BatchNormLayer(D=feature_maps, name='conv1_2/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_2/activation')(x)

        x = ConvLayer(
            kw=3, kh=3, in_f=feature_maps, out_f=output_factorization_layer,
            use_bias=use_bias, activation=None, name='conv1_3/weights',
            kernel_initializer=kernel_initializer
        )(x)
        x = BatchNormLayer(D=output_factorization_layer, name='conv1_3/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_3/activation')(x)

        feature_maps = output_factorization_layer
    elif using_zero_padding:
        if head_bn_params is None:
            head_bn_params = get_head_batchnorm_params()

        x = BatchNormLayer(D=input_shape[-1], name='bn_data', **head_bn_params)(in_x)

        x = ZeroPaddingLayer(padding=[[3, 3], [3, 3]], name='zero_padding2d')(x)
        
        x = ConvLayer(
            kw=7, kh=7, in_f=input_shape[-1], out_f=feature_maps, stride=stride_list[0],
            use_bias=False, activation=None, padding='VALID',name='conv0',
            kernel_initializer=kernel_initializer
        )(x)
        
        x = BatchNormLayer(D=feature_maps, name='bn0', **bn_params)(x)
        x = ActivationLayer(name='activation0')(x)

        x = ZeroPaddingLayer(padding=[[1, 1], [1, 1]], name='zero_padding2d_1')(x)
    else:
        x = ConvLayer(
            kw=7, kh=7, in_f=input_shape[-1], out_f=feature_maps, use_bias=use_bias,
            stride=stride_list[0], activation=None,name='conv1/weights',
            kernel_initializer=kernel_initializer
        )(in_x)
        x = BatchNormLayer(D=feature_maps, name='conv1/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='activation')(x)

    if using_zero_padding:
        x = MaxPoolLayer(
            strides=[1, stride_list[1], stride_list[1], 1],
            ksize=[1,3,3,1],
            padding='VALID',
            name='max_pooling2d'
        )(x)
    else:
        x = MaxPoolLayer(strides=[1, stride_list[1], stride_list[1], 1], ksize=[1, 3, 3, 1], name='max_pooling2d')(x)

    # Build body of ResNet
    num_activation = 3
    num_block = 0
    num_stride = 2

    for stage, repeat in enumerate(repetition):

        # All stages begins at 1 more
        stage += 1

        for block in range(1, repeat + 1):

            # First block of the first stage is used without strides because we have maxpooling before
            if block == 1 and stage == 1:
                if pointwise:
                    x = conv_block(
                        x=x, 
                        block_id=stage, 
                        unit_id=block, 
                        num_block=num_block,
                        use_bias=use_bias,
                        activation=activation,
                        stride=1,
                        out_f=256,
                        reduction=min_reduction,
                        kernel_initializer=kernel_initializer,
                        bn_params=bn_params
                    )
                else:
                    x = conv_block(
                        x=x,
                        block_id=stage, 
                        unit_id=block, 
                        num_block=num_block,
                        use_bias=use_bias,
                        activation=activation,
                        stride=1,
                        out_f=feature_maps,
                        kernel_initializer=kernel_initializer,
                        bn_params=bn_params
                    )
            elif block == 1:
                # Every first block in new stage (zero block) we do block with stride 2 and increase number of feature maps
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=stride_list[num_stride],
                    kernel_initializer=kernel_initializer,
                    bn_params=bn_params
                )
                num_stride += 1
            else:
                x = iden_block(
                    x=x,
                    block_id=stage,
                    unit_id=block,
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bn_params=bn_params
                )
            num_block += 1

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
    
    if not pointwise:
        x = BatchNormLayer(D=x.get_shape()[-1], name='bn1', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='relu1')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_classes,
            activation=None, name='logits' if pointwise else 'fc1'
        )(x)

        if create_model:
            return Classificator(in_x, output, name=name_model)
    else:
        output = x

    return in_x, output


def build_LittleResNetV1(
        input_shape,
        depth=20,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=False,
        name_model='MakiClassificator',
        activation_between_blocks=True,
        kernel_initializer=InitConvKernel.HE,
        input_tensor=None):
    """
    These type of ResNet tests on CIFAR-10 and CIFAR-100

    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    depth : int
        Maximum number of layers.
    use_bias : bool
        If true, when on layers used bias operation.
    activation : tf object
        Activation on every convolution layer.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    activation_between_blocks : bool
        Use activation between blocks.
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.

    Returns
    ---------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """

    feature_maps = 16
    bm_params = get_batchnorm_params()

    conv_block = ResNetConvBlock_woPointWiseV1
    iden_block = ResNetIdentityBlock_woPointWiseV1

    if input_tensor is None:
        in_x = InputLayer(input_shape=input_shape, name='Input')
    elif input_tensor is not None:
        in_x = input_tensor
        input_shape = input_tensor.get_shape()

    x = ConvLayer(
        kw=3, kh=3, in_f=input_shape[-1], out_f=feature_maps, activation=None,
        use_bias=use_bias, name='conv1',
        kernel_initializer=kernel_initializer
    )(in_x)
    x = BatchNormLayer(D=feature_maps, name='bn_1', **bm_params)(x)
    x = ActivationLayer(activation=activation, name= 'activation_1')(x)

    repeat = int((depth - 2) / 6)

    # Build body of ResNet
    num_block = 0
    num_activation = 3
    
    for stage in range(3):
        for block in range(1, repeat + 1):

            # All stages begins at 1 more
            stage += 1

            # First block of the first stage is used without strides because we have maxpooling before
            if block == 1 and stage == 1:
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=1,
                    out_f=feature_maps,
                    kernel_initializer=kernel_initializer,
                    bn_params=bm_params
                )
            elif block == 1:
                # Every first block in new stage (zero block) we do block with stride 2 and increase number of feature maps
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=2,
                    kernel_initializer=kernel_initializer,
                    bn_params=bm_params
                )
            else:
                x = iden_block(
                    x=x,
                    block_id=stage,
                    unit_id=block,
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bn_params=bm_params
                )

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
            num_block += 1

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(in_d=x.get_shape()[-1], out_d=num_classes, activation=None, name='logits')(x)
    else:
        output = x

    if create_model:
        return Classificator(in_x,output,name=name_model)
    else:
        return in_x, output

