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

from makiflow.layers import *
from makiflow.old.models import Classificator
import numpy as np
import tensorflow as tf


def classificator(input_shape=[32, 32, 3], n_classes=10, train_batch_size=None):
    """
    Creates a Classificator model.

    Parameters
    ----------
    input_shape : list
        Contains the shape of the input tensor. Must not include the batch dimension.
    n_classes : int
        The number of classes.
    train_batch_size : int
        If provided, an InputLayer with same name and input_shape will be returned, but the
        batch dimension will be set to `train_batch_size`.

    Returns
    -------
    Classificator
        The classificator model.
    InputLayer
        Training input layer with the specified `train_batch_size` batch size.
        The input layer is not returned if `train_batch_size` is not provided.
    """
    in_x = InputLayer([1, *input_shape], name='input_image')
    x = ConvLayer(kw=3, kh=3, in_f=3, out_f=32, stride=1, name='conv1')(in_x)

    conv_id = 2
    while x.get_shape()[1] > 3:
        x = ConvLayer(kw=3, kh=3, in_f=32, out_f=32, stride=2, name=f'conv{conv_id}')(x)
        conv_id += 1

    x = FlattenLayer('flatten')(x)
    out_d = x.get_shape()[-1]
    out_x = DenseLayer(in_d=out_d, out_d=n_classes, activation=None, name='class_head')(x)
    model = Classificator(in_x=in_x, out_x=out_x)

    if train_batch_size is not None:
        input_shape = [train_batch_size, *input_shape]
        train_in_x = InputLayer(input_shape, name='input_image')
        return model, train_in_x

    return model


if __name__ == '__main__':
    print('Building model...')
    model = classificator()
    print('Setting the session...')
    model.set_session(tf.Session())
    print('Predicting...')
    answer = model.predict(np.random.randn(1, 32, 32, 3))
    print(f'answer[0]={answer[0]}')
    error_rate = model.evaluate(np.random.randn(1, 32, 32, 3), [1])
    print('error_rate', error_rate)
    print('Building model + training layer...')
    model, train_in_x = classificator(train_batch_size=32)
    print('Built model:', model)
    print('Training input layer', train_in_x)
    print(model.get_feed_dict_config())
