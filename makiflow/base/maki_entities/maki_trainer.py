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

from abc import ABC
from makiflow.base.maki_entities.maki_model import MakiModel
from makiflow.base.maki_entities.maki_layer import MakiRestorable
from .maki_tensor import MakiTensor
import tensorflow as tf
from copy import copy


class MakiTrainer(MakiModel, ABC):
    # Provides API for training the model.
    def __init__(self, graph_tensors: dict, outputs: list, inputs: list):
        self._set_for_training = False
        super().__init__(graph_tensors, outputs, inputs)

    def _setup_for_training(self):
        self._set_for_training = True

        # Collect all the layers names since all of them are trainable from
        # the beginning.
        self._trainable_layers = []
        for layer_name in self._graph_tensors:
            self._trainable_layers.append(layer_name)
        self._trainable_vars = []
        self._collect_train_params()

        # Setup L2 regularization
        self._uses_l2_regularization = False
        self._l2_reg_loss_is_build = False
        self._l2_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l2_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

        # Setup L1 regularization
        self._uses_l1_regularization = False
        self._l1_reg_loss_is_build = False
        self._l1_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l1_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

        # Setup external loss
        self._uses_external_loss = False

    def set_layers_trainable(self, layers):
        """

        Parameters
        ----------
        layers: list
            List of tuples: (layer_name, bool).
            Example: to make layer called `conv1` untrainable use
            the following tuple (`conv1`, False), if you want to make it trainable
            use the following tuple (`conv1`, True).
        """
        if not self._set_for_training:
            self._setup_for_training()
        for layer_name, is_trainable in layers:
            if is_trainable and layer_name not in self._trainable_layers:
                self._trainable_layers.append(layer_name)
            elif not is_trainable and layer_name in self._trainable_layers:
                self._trainable_layers.remove(layer_name)
        # It will collect the trainable parameters and rebuild the graph.
        # Graph rebuild is needed since some of the layers behave differently in
        # training mode.
        self._collect_train_params()

    def _collect_train_params(self):
        self._trainable_vars.clear()
        for layer_name in self._trainable_layers:
            layer = self._graph_tensors[layer_name].get_parent_layer()
            self._trainable_vars += layer.get_params()
        # Create graph or refresh it
        self._build_training_graph()

    # L2 REGULARIZATION
    def set_l2_reg(self, layers):
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
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True

        for layer_name, decay in layers:
            self._l2_regularized_layers[layer_name] = decay

    def set_common_l2_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training.
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l2_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True

        for layer_name in self._l2_regularized_layers:
            if self._l2_regularized_layers[layer_name] is not None:
                self._l2_regularized_layers[layer_name] = decay

    def _build_l2_loss(self):
        self._l2_reg_loss = tf.constant(0.0)
        for layer_name in self._l2_regularized_layers:
            decay = self._l2_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params_regularize()
                for param in params:
                    self._l2_reg_loss += tf.nn.l2_loss(param) * tf.constant(decay)

        self._l2_reg_loss_is_build = True

    # L1 REGULARIZATION

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
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True

        for layer_name, decay in layers:
            self._l1_regularized_layers[layer_name] = decay

    def set_common_l1_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training.
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l1_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True

        for layer_name in self._l1_regularized_layers:
            if self._l1_regularized_layers[layer_name] is not None:
                self._l1_regularized_layers[layer_name] = decay

    def _build_l1_loss(self):
        self._l1_reg_loss = tf.constant(0.0)
        for layer_name in self._l1_regularized_layers:
            decay = self._l1_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params_regularize()
                for param in params:
                    self._l1_reg_loss += tf.abs(tf.reduce_sum(param)) * tf.constant(decay)

        self._l1_reg_loss_is_build = True

    # noinspection PyAttributeOutsideInit
    def add_loss(self, loss, scale=1.0):
        """
        Adds an external loss that is defined outside the model or trainer.
        Can be used for such things as perceptual loss.
        Parameters
        ----------
        loss : tf.Tensor
            A scalar that will be added to all the other losses used to train the model.
        scale : float
            A scalar by which the loss will be scaled.
        """
        # noinspection PyTypeChecker
        self._external_loss = loss * scale
        self._uses_external_loss = True

    def _build_final_loss(self, training_loss):
        # Adds regularization terms to the actual loss.
        if self._uses_l1_regularization:
            if not self._l1_reg_loss_is_build:
                self._build_l1_loss()
            training_loss += self._l1_reg_loss

        if self._uses_l2_regularization:
            if not self._l2_reg_loss_is_build:
                self._build_l2_loss()
            training_loss += self._l2_reg_loss

        if self._uses_external_loss:
            training_loss += self._external_loss

        return training_loss

    def _build_training_graph(self):
        # The algorithm recursively goes down the graph until it finds the input layer
        # and then passes its tensor through all the layers it has encountered so far.

        # Contains pairs {layer_name: {MakiTensor name: data_tensor}}, where the inner dictionary
        # contains all the MakiTensor that were produced by the layer with the `layer_name` name.
        output_tensors = {}

        def create_tensor(maki_tensor: MakiTensor):
            # Check if the parent layer has been already used.
            # If it has, the required tensor has already been constructed.
            layer = maki_tensor.get_parent_layer()

            # Check if we haven't used this layer before.
            # If haven't, add an empty dictionary.
            if output_tensors.get(layer.get_name()) is None:
                output_tensors[layer.get_name()] = dict()

            outputs = output_tensors.get(layer.get_name())
            # Check if the tensor has already been created.
            if outputs.get(maki_tensor.get_name()) is not None:
                return outputs.get(maki_tensor.get_name())

            # If we are here, then the tensor hasn't been created.
            # Check if we at the beginning of the computational graph, i.e. InputLayer
            if len(maki_tensor.get_parent_tensor_names()) == 0:
                X = maki_tensor.get_data_tensor()
                outputs.update(
                    {maki_tensor.get_name(): X}
                )
                return X

            # Collect tensors that were used to create current `maki_tensor`
            parent_tensors = []
            for tensor in maki_tensor.get_parent_tensors():
                parent_tensors += [create_tensor(tensor)]

            # If only one tensor is used for creation, then the layer does not expect
            # a list.
            if len(parent_tensors) == 1:
                parent_tensors = parent_tensors[0]

            if layer.get_name in self._trainable_layers:
                X = layer._training_forward(
                    parent_tensors
                )
            else:
                X = layer._forward(
                    parent_tensors,
                    computation_mode=MakiRestorable.TRAINING_MODE
                )

            # Check if the layer outputs several tensors.
            # If not, put the returned tensor into a list.
            if not isinstance(X, list):
                X = [X]

            # Get names of the MakiTensors that were created
            # after passing parent of the `maki_tensor` through the `layer`.
            # Order of the names is always the same as the order
            # of the returned tensors.
            # This is done this way because the same layer can be reused several times.
            parent_name = maki_tensor.get_parent_tensor_names()[0]
            output_names = layer.get_children(parent_name)
            for _x, x_name in zip(X, output_names):
                outputs.update({x_name: _x})

            return outputs.get(maki_tensor.get_name())

        for output in self._outputs:
            self._training_outputs += [create_tensor(output)]
