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

from .maki_core import MakiCore
from ..graph_entities import MakiTensor
from makiflow.core.debug import ExceptionScope
import tensorflow as tf
from abc import abstractmethod
import json


class ModelSerializer(MakiCore):
    MODEL_INFO = 'model_info'
    GRAPH_INFO = 'graph_info'

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
        ] + [
            single_output.get_data_tensor().name.split(':')[0]
            for single_output in super().get_graph_tensors().values()
        ]

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

    def freeze_graph(self, output_tensors: list = None):
        """
        Creates a frozen instance of the model's computational graph.

        Parameters
        ----------
        output_tensors : list
            List of MakiTensors or tf.Tensors that resemble the output tensors of interest.

        Returns
        -------
        GraphDef
            Frozen graph definition.
        """
        with ExceptionScope('MakiModel.freeze_graph'):
            assert super().get_session() is not None, 'The model must be initialized with a session.'

            t_name = lambda x: x.split(':')[0]

            if output_tensors is None:
                print('Output tensors are not provided. Using the standard ones:')
                print(super().get_outputs())
                print("Use `get_outputs` in order to obtain the output tensors and their names.")
                output_tensors = super().get_outputs()

            # output_tensors contains MakiTensors
            if isinstance(output_tensors[0], MakiTensor):
                output_names = [t_name(x.get_data_tensor().name) for x in output_tensors]
            # output_tensors contains tf.Tensors
            else:
                output_names = [t_name(x.name) for x in output_tensors]

            session = super().get_session()
            graph = session.graph
            with graph.as_default():
                # Collect model parameters' names
                var_names = [x.op.name for _, x in self._named_dict_params.items()]
                graph_def = graph.as_graph_def()

                # Create the frozen graph entity
                frozen_graph = tf.graph_util.convert_variables_to_constants(
                    sess=session,
                    input_graph_def=graph_def,
                    output_node_names=output_names,
                    variable_names_whitelist=var_names
                )

            return frozen_graph

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
            ModelSerializer.MODEL_INFO: model_info,
            ModelSerializer.GRAPH_INFO: graph_info
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

    @staticmethod
    @abstractmethod
    def from_json(path: str, input_tensor: MakiTensor = None):
        pass

    @staticmethod
    def load_architecture(path):
        """
        Opens json file at the given path and returns its contents.

        Parameters
        ----------
        path : str
            Path to the json to read.

        Returns
        -------
        dict
            The contents of the json.
        """
        with open(path) as file:
            data = json.load(file)

        model_info = data[ModelSerializer.MODEL_INFO]
        graph_info = data[ModelSerializer.GRAPH_INFO]
        return model_info, graph_info
