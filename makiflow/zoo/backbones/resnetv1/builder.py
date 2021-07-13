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

from .blocks import identity_block as with_pointwise_IB
from .blocks import conv_block as with_pointwise_CB

from .blocks import without_pointwise_IB
from .blocks import without_pointwise_CB

from .utils import get_batchnorm_params, get_batchnorm_params_resnet34

from makiflow.layers import *
from makiflow import Model, MakiTensor


def build_ResNetV1(
        in_x: MakiTensor,
        repetition=(2, 2, 2, 2),
        include_top=False,
        num_classes=1000,
        factorization_first_layer=False,
        use_bias=False,
        using_zero_padding=False,
        stride_list=(2, 2, 2, 2, 2),
        head_bn_params={},
        activation=tf.nn.relu,
        block_type='with_pointwise',
        create_model=False,
        name_model='MakiClassificator',
        init_filters=64,
        min_reduction=64,
        activation_between_blocks=True,
        input_tensor=None):
    """
    Parameters
    ----------
    in_x : MakiTensor
        A tensor that will be fed into the model.
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
    input_tensor : mf.MakiTensor

    Returns
    ---------
    in_x :  mf.MakiTensor
        Input  mf.MakiTensor.
    output : int
        Output mf.MakiTensor.
    Classificator : mf.models.Model
        Constructed model
    """

    if (type(repetition) is not list and type(repetition) is not tuple) or len(repetition) != 4:
        raise TypeError('repetition should be list of size 4')

    feature_maps = init_filters
    if using_zero_padding:
        bn_params = get_batchnorm_params_resnet34()
    else:
        bn_params = get_batchnorm_params()

    if block_type == 'with_pointwise':
        conv_block = with_pointwise_CB
        iden_block = with_pointwise_IB
        output_factorization_layer = init_filters
        pointwise = True
    elif block_type == 'without_pointwise':
        conv_block = without_pointwise_CB
        iden_block = without_pointwise_IB
        output_factorization_layer = init_filters * 2
        pointwise = False
    else:
        raise Exception(f'{block_type} type is not found')

    if factorization_first_layer:

        x = ConvLayer(kw=3, kh=3, in_f=in_x.shape[-1], out_f=feature_maps, use_bias=use_bias,
                      activation=None, name='conv1_1/weights')(in_x)

        x = BatchNormLayer(D=feature_maps, name='conv1_1/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_1/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=feature_maps, use_bias=use_bias,
                      activation=None, name='conv1_2/weights')(x)

        x = BatchNormLayer(D=feature_maps, name='conv1_2/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_2/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=output_factorization_layer,
                      use_bias=use_bias, stride=stride_list[0], activation=None, name='conv1_3/weights')(x)

        x = BatchNormLayer(D=output_factorization_layer, name='conv1_3/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_3/activation')(x)

        feature_maps = output_factorization_layer
    elif using_zero_padding:
        x = BatchNormLayer(D=in_x.shape[-1], name='bn_data', **head_bn_params)(in_x)

        x = ZeroPaddingLayer(padding=[[3, 3], [3, 3]], name='zero_padding2d')(x)
        x = ConvLayer(kw=7, kh=7, in_f=in_x.shape[-1],
                      out_f=feature_maps, stride=stride_list[0], use_bias=False, activation=None, padding='VALID',
                      name='conv0')(x)
        x = BatchNormLayer(D=feature_maps, name='bn0', **bn_params)(x)
        x = ActivationLayer(name='activation0')(x)

        x = ZeroPaddingLayer(padding=[[1, 1], [1, 1]], name='zero_padding2d_1')(x)
    else:
        x = ConvLayer(kw=7, kh=7, in_f=in_x.shape[-1], out_f=feature_maps, use_bias=use_bias,
                      stride=stride_list[0], activation=None, name='conv1/weights')(in_x)

        x = BatchNormLayer(D=feature_maps, name='conv1/BatchNorm', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='activation')(x)

    if using_zero_padding:
        x = MaxPoolLayer(
            strides=[1, stride_list[1], stride_list[1], 1],
            ksize=[1, 3, 3, 1],
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
                    bn_params=bn_params
                )
            num_block += 1

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3

    if not pointwise:
        x = BatchNormLayer(D=x.shape[-1], name='bn1', **bn_params)(x)
        x = ActivationLayer(activation=activation, name='relu1')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(in_d=x.shape[-1], out_d=num_classes, activation=None,
                            name='logits' if pointwise else 'fc1')(x)
    else:
        output = x

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)

    return output


def build_LittleResNetV1(
        in_x: MakiTensor,
        depth=20,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=False,
        name_model='MakiClassificator',
        activation_between_blocks=True,
        input_tensor=None
):
    """
    These type of ResNet tests on CIFAR-10 and CIFAR-100

    Parameters
    ----------
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    depth : list
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
    input_tensor : mf.MakiTensor
        A tensor that will be fed into the model instead of InputLayer with the specified `input_shape`.

    Returns
    ---------
    in_x :  mf.MakiTensor
        Input MakiTensor.
    output : int
        Output MakiTensor.
    Classificator : mf.models.Model
        Constructed model.
    """

    feature_maps = 16
    bm_params = get_batchnorm_params()

    conv_block = without_pointwise_CB
    iden_block = without_pointwise_IB

    x = ConvLayer(kw=3, kh=3, in_f=in_x.shape[-1], out_f=feature_maps, activation=None,
                  use_bias=use_bias, name='conv1')(in_x)

    x = BatchNormLayer(D=feature_maps, name='bn_1', **bm_params)(x)
    x = ActivationLayer(activation=activation, name='activation_1')(x)

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
                    bn_params=bm_params
                )

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
            num_block += 1

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(in_d=x.shape[-1], out_d=num_classes, activation=None, name='logits')(x)
    else:
        output = x

    if create_model:
        return Model(inputs=in_x, outputs=output, name=name_model)
    else:
        return output
