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

from abc import abstractmethod, ABC
import tensorflow as tf
import json


class MakiModel(ABC):
    # Provides basic API for all the models:
    # saving/loading weights;
    # saving model architecture;
    # encapsulates all the graph info;
    # encapsulates session.

    MODEL_INFO = 'model_info'
    GRAPH_INFO = 'graph_info'

    def __init__(self, graph_tensors: dict, outputs: list, inputs: list):
        # Contains all the MakiTensor that appear in the computational graph
        self._graph_tensors = graph_tensors
        self._outputs = outputs
        self._inputs = inputs
        self._session = None

        # Extracted output tf.Tensors
        self._output_data_tensors = []
        for maki_tensor in self._outputs:
            self._output_data_tensors += [maki_tensor.get_data_tensor()]

        # Extracted input tf.Tensors
        self._input_data_tensors = []
        for maki_tensor in self._inputs:
            self._input_data_tensors += [maki_tensor.get_data_tensor()]

        self._collect_params()

    def _collect_params(self):
        self._params = []
        self._named_dict_params = {}
        for tensor_name in self._graph_tensors:
            layer = self._graph_tensors[tensor_name].get_parent_layer()
            self._params += layer.get_params()
            self._named_dict_params.update(layer.get_params_dict())

    def set_session(self, session: tf.Session):
        self._session = session
        params = []
        # Do not initialize variables using self._params since
        # self._params contains only gradient descent trainable parameters.
        # It is not the case with BatchNormalization where mean and variance are
        # computed on the fly.
        for key in self._named_dict_params:
            params += [self._named_dict_params[key]]
        init_op = tf.variables_initializer(params)
        self._session.run(init_op)

    def load_weights(self, path, layer_names=None):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        Example: '/home/student401/my_model/model.ckpt'
        Parameters
        ----------
            path : str
                Full path to stored weights
            layer_names : list of str
                Names of layer which weights need load from file into model
        """
        vars_to_load = {}
        if layer_names is not None:
            for layer_name in layer_names:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                vars_to_load.update(layer.get_params_dict())
        else:
            vars_to_load = self._named_dict_params
        saver = tf.train.Saver(vars_to_load)
        saver.restore(self._session, path)
        print('Weights are loaded.')

    def save_weights(self, path, layer_names=None):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        Example: '/home/student401/my_model/model.ckpt'
        Parameters
        ----------
            path : str
                Full path to place where weights should saved
            layer_names : list of str
                Names of layer which weights need save from model
        """
        vars_to_save = {}
        if layer_names is not None:
            for layer_name in layer_names:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                vars_to_save.update(layer.get_params_dict())
        else:
            vars_to_save = self._named_dict_params
        saver = tf.train.Saver(vars_to_save)
        save_path = saver.save(self._session, path)
        print(f'Weights are saved to {save_path}')

    def save_model_as_pb(self, path_to_save: str, file_name: str):
        """
        Save model (i. e. tensorflow graph) as pb (protobuf) file,
        which is convenient to use in the future

        Parameters
        ----------
        path_to_save : str
            Path to save model,
            For example: /home/user_name/weights/
        file_name : str
            Name of the file which is pb file,
            For example: model.pb
        """
        # Collect all names of output variables/operations
        output_names = [
            single_output.get_data_tensor().name.split(':')[0]
            for single_output in self._outputs
        ]

        frozen_graph = None
        graph = self._session.graph
        with graph.as_default():
            # Collects all names of variables which are need to freeze
            freeze_var_names = [v.op.name for v in tf.global_variables()]
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                self._session, input_graph_def, output_names, freeze_var_names
            )

        tf.train.write_graph(frozen_graph, path_to_save,
                             file_name, as_text=False
        )

    def save_architecture(self, path):
        """
        This function save architecture of model in certain path.
        Example: '/home/student401/my_model/architecture.json'
        Parameters
        ----------
            path : str
                Full path to place where architecture should saved
        """
        model_info = self._get_model_info()
        graph_info = self._get_graph_info()
        model_dict = {
            MakiModel.MODEL_INFO: model_info,
            MakiModel.GRAPH_INFO: graph_info
        }

        model_json = json.dumps(model_dict, indent=1)
        json_file = open(path, mode='w')
        json_file.write(model_json)
        json_file.close()
        print(f"Model's architecture is saved to {path}.")

    @abstractmethod
    def _get_model_info(self):
        """
        This method must be implemented by other models.
        """
        pass

    def _get_graph_info(self):
        tensor_dicts = []
        for tensor_name in self._graph_tensors:
            tensor = self._graph_tensors[tensor_name]
            tensor_dicts += [tensor.to_dict()]
        return tensor_dicts

    def get_node(self, node_name):
        return self._graph_tensors.get(node_name)
