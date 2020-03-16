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

from tensorflow.nn import relu, sigmoid, tanh, softmax, leaky_relu, relu6


class ActivationConverter:
    """
    This class is used for saving models. We need to know what activation
    function is used in a particular layer, so we map them with some string values 
    like tensorflow.nn.relu -> relu. This way we can then recover the model cause we
    know what activation function is used.
    """

    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LEAKY_RELU = 'leaky_relu'
    RELU6 = 'relu6'
    NONE = 'None'

    @staticmethod
    def activation_to_str(activation):
        return {
            relu: ActivationConverter.RELU,
            sigmoid: ActivationConverter.SIGMOID,
            tanh: ActivationConverter.TANH,
            softmax: ActivationConverter.SOFTMAX,
            leaky_relu: ActivationConverter.LEAKY_RELU,
            relu6: ActivationConverter.RELU6,
            None: ActivationConverter.NONE
        }[activation]

    @staticmethod
    def str_to_activation(activation_name):
        return {
            ActivationConverter.RELU: relu,
            ActivationConverter.SIGMOID: sigmoid,
            ActivationConverter.TANH: tanh,
            ActivationConverter.SOFTMAX: softmax,
            ActivationConverter.LEAKY_RELU: leaky_relu,
            ActivationConverter.RELU6: relu6,
            ActivationConverter.NONE: None
        }[activation_name]

