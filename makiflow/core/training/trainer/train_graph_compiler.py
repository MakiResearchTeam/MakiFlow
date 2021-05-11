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

from makiflow.core.graph_entities import MakiTensor, MakiRestorable
from makiflow.core.inference.maki_core import MakiCore
from ..core import TensorProvider


class GraphCompiler(TensorProvider):
    # This entity is responsible for building the training graph and
    # the final loss
    def __init__(self, model: MakiCore, train_inputs: list):
        """
        Provides basic tools for the training setup. Builds final loss tensor and the training graph.
        Parameters
        ----------
        model : MakiCore
            The model's object.
        train_inputs : list
            List of the input training tensors. Their names must be the same as their inference counterparts!
        """
        self._model = model
        self._graph_tensors = model.get_graph_tensors()
        self._train_inputs_list = train_inputs
        self._train_inputs = {}
        for train_input in train_inputs:
            self._train_inputs.update(train_input.get_self_pair())

        self._is_compiled = False
        self._init()

    def get_train_inputs_list(self):
        return self._train_inputs_list.copy()

    def get_session(self):
        return self._model.get_session()

    def get_model(self):
        return self._model

    def get_batch_size(self):
        """
        Returns
        -------
        int
            The batch size. It uses the first shape dimension of the first input MakiTensor.
        """
        return self._train_inputs_list[0].shape()[0]

    def compile(self):
        """
        Initiates building the training graph. Has to be called before the training.
        """
        print('Compile the model...')
        self.compile_training_graph()
        self._is_compiled = True
        print('Model is compiled.')

    def is_compiled(self):
        return self._is_compiled

    def _init(self):
        """
        This method must be used by other gyms to create necessary variables.
        The parent's `_init` must be called first.
        """
        # Collect all the layers names since all of them are trainable from
        # the beginning.
        self._trainable_layers = []
        for layer_name in self._graph_tensors:
            self._trainable_layers.append(layer_name)
        self._trainable_vars = []
        self._collect_train_params()

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
            layer = self._graph_tensors[layer_name].parent_layer
            self._trainable_vars += layer.get_params()

    def get_trainable_params(self):
        return self._trainable_vars

    def compile_training_graph(self):
        # The algorithm recursively goes down the graph until it finds the input layer
        # and then passes its tensor through all the layers it has encountered so far.

        # Contains pairs {layer_name: {MakiTensor name: data_tensor}}, where the inner dictionary
        # contains all the MakiTensor that were produced by the layer with the `layer_name` name.
        layer_name2output_tensors = {}
        # Collection of all the tf.Tensor that stem from the training graph.
        self._traingraph_tensors = {}

        def create_tensor(maki_tensor: MakiTensor):
            # If the parent layer has been used, the required tensor is already constructed.
            layer = maki_tensor.parent_layer

            # Check if we haven't used this layer before.
            # If haven't, add an empty dictionary.
            if layer_name2output_tensors.get(layer.name) is None:
                layer_name2output_tensors[layer.name] = dict()

            outputs = layer_name2output_tensors.get(layer.name)

            # Check if the tensor has already been created.
            if outputs.get(maki_tensor.name) is not None:
                return outputs.get(maki_tensor.name)

            # If we are here, then the tensor hasn't been created.

            # Check if we at the beginning of the computational graph, i.e. InputLayer
            if len(maki_tensor.parent_tensor_names) == 0:
                # Replace an inference input tensor with its training counterpart
                name = maki_tensor.name
                training_makitensor = self._train_inputs.get(name)
                if training_makitensor is None:
                    raise KeyError(f'There is no training input tensor with name {name}. The names of the training'
                                   f'input tensors must be the same with their corresponding inference counterparts.')

                X = training_makitensor.tensor
                outputs.update(
                    {maki_tensor.name: X}
                )
                self._traingraph_tensors.update(
                    {name: X}
                )
                return X

            # Collect tensors that were used to create current `maki_tensor`
            parent_tensors = []
            for tensor in maki_tensor.parent_tensors:
                parent_tensors += [create_tensor(tensor)]

            # If only one tensor is used for creation, then the layer does not expect
            # a list.
            if len(parent_tensors) == 1:
                parent_tensors = parent_tensors[0]

            if layer.name in self._trainable_layers:
                X = layer.training_forward(
                    parent_tensors
                )
            else:
                X = layer.forward(
                    parent_tensors,
                    computation_mode=MakiRestorable.TRAINING_MODE
                )

            # Check if the layer outputs several tensors.
            # If not, put the returned tensor into a list.
            if not isinstance(X, tuple):
                X = [X]

            # Get names of the MakiTensors that were created
            # after passing parent of the `maki_tensor` through the `layer`.
            # Order of the names is always the same as the order
            # of the returned tensors.
            # This is done this way because the same layer can be reused several times.
            parent_name = maki_tensor.parent_tensor_names[0]
            output_names = layer.get_children(parent_name)
            for _x, x_name in zip(X, output_names):
                outputs.update({x_name: _x})
                self._traingraph_tensors[x_name] = _x

            return outputs.get(maki_tensor.name)

        for output in self._model.get_outputs():
            # Even though the method does return some tensors, they are not being collected here.
            # It is done internally in the method. All the necessary tensors can be accessed
            # via the `get_traingraph_tensor` method.
            create_tensor(output)

        self._is_compiled = True

    def get_traingraph_tensor(self, tensor_name):
        """
        Returns a datatensor of a MakiTensor with the specified `tensor_name`.
        Parameters
        ----------
        tensor_name : str
            Name of the MakiTensor which datatensor to get.
        Returns
        -------
        tf.Tensor
        """
        tensor = self._traingraph_tensors.get(tensor_name)
        if tensor is None:
            raise KeyError(f'Could not find training tensor with name={tensor_name}')
        return tensor
