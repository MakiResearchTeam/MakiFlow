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

import tensorflow as tf
from .aion import Aion
from abc import ABC


class L1(Aion, ABC):
    # L1 REGULARIZATION
    def _init(self):
        super()._init()
        # Setup L1 regularization
        self._uses_l1_regularization = False
        self._l1_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l1_regularized_layers[layer_name] = None

    def set_l1_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.
        WARNING! It is assumed that `set_layers_trainable` method won't
        be used anymore.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        # noinspection PyAttributeOutsideInit
        self._uses_l1_regularization = True

        for layer_name, decay in layers:
            self._l1_regularized_layers[layer_name] = decay

    def set_common_l1_weight_decay(self, decay=1e-6):
        """
        Sets `decay` for all trainable weights (that can be regularized).

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        # noinspection PyAttributeOutsideInit
        self._uses_l1_regularization = True

        for layer_name in self._l1_regularized_layers:
            self._l1_regularized_layers[layer_name] = decay

    def __build_l1_loss(self):
        self._l1_reg_loss = tf.constant(0.0)
        for layer_name in self._l1_regularized_layers:
            decay = self._l1_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params_regularize()
                for param in params:
                    self._l1_reg_loss += tf.reduce_sum(tf.abs(param)) * tf.constant(decay)

        self._l1_reg_loss_is_build = True

    def _build_final_loss(self, training_loss):
        if self._uses_l1_regularization:
            self.__build_l1_loss()
            training_loss += self._l1_reg_loss

        return super()._build_final_loss(training_loss)

    def get_l1_regularization_loss(self):
        assert self._l1_reg_loss_is_build, 'The loss has not been built yet. Please compile the model'
        return self._l1_reg_loss
