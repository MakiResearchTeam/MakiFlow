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

from abc import abstractmethod
from .maki_layer import MakiRestorable
from .maki_tensor import MakiTensor
from .input_maki_layer import InputMakiLayer


class MakiBuilder:
    # Provides API for model restoration.

    # Contains pairs {LayerType: LayerClass}.
    # This static field gets filled within the layers packages.
    ALL_LAYERS_ADDRESS = {}

    @staticmethod
    def register_layers(type_layer_dict):
        MakiBuilder.ALL_LAYERS_ADDRESS.update(type_layer_dict)

    @staticmethod
    def __layer_from_dict(layer_dict):
        """
        Creates and returns Layer from dictionary
        """
        params = layer_dict[MakiRestorable.PARAMS]

        build_layer = MakiBuilder.ALL_LAYERS_ADDRESS.get(layer_dict[MakiRestorable.FIELD_TYPE])

        if build_layer is None:
            raise KeyError(f'{layer_dict[MakiRestorable.FIELD_TYPE]} was not found!')

        return build_layer.build(params)

    # -----------------------------------------------------------GRAPH RESTORATION--------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def restore_graph(outputs, graph_info_json, input_layer: InputMakiLayer = None):
        """
        Restore Inference graph with inputs and outputs of model from json.

        Parameters
        ----------
        outputs : list
            List of the names of the output MakiTensors of the model.
        graph_info_json : dict
            Graph info section from the architecture file.
        input_layer : InputMakiLayer
            Custom InputLayer. Use this parameter if you want to train the model with pipelines
            or simply want to change the batch size.

        Returns
        -------
        dict
            Contains all the MakiTensors that appear in the graph before `outputs`, including
            the `outputs` MakiTensors.
        """
        # dict {makitensor_name : {
        #           name: makitensor_name,
        #           parent_layer_info: {...},
        #           parent_tensor_names: [...]
        # }
        # Used for fast querying of the MakiTensors during graph restoration.
        graph_info = {}
        for tensor in graph_info_json:
            graph_info[tensor[MakiRestorable.NAME]] = tensor

        # Collects all the created MakiTensors.
        # Contains pairs {makitensor_name: MakiTensor}.
        makitensors = {}
        # Collects all the created layers.
        # Contains pairs {layer_name: MakiLayer}.
        layers = {}

        def restore_makitensor(makitensor_name):
            """
            Restores the requested MakiTensor.
            """
            makitensor_info = graph_info[makitensor_name]

            # Check if the makitensor was already created.
            if makitensors.get(makitensor_name) is not None:
                return makitensors[makitensor_name]

            parent_makitensor_names = makitensor_info[MakiTensor.PARENT_TENSOR_NAMES]
            # Check if we at the beginning of the graph. In this case we create InputLayer and return it.
            if len(parent_makitensor_names) == 0:
                layer = get_parent_layer(makitensor_info[MakiTensor.PARENT_LAYER_INFO], layer=input_layer)
                # If a custom InputLayer is passed, then it may have a different name. It this case
                # we need to update parent tensors names. If no inputlayer is passed, the code below
                # will change nothing.
                return layer

            parent_makitensors = []
            for parent_makitensor_name in parent_makitensor_names:
                # This call may modify `graph_info` if a custom InputLayer is passed.
                parent_makitensors += [restore_makitensor(parent_makitensor_name)]

            # If a custom InputLayer is passed, then it may have a different name. It this case
            # we need to update parent tensors names. If no InputLayer is passed, the code below
            # will change nothing.
            parent_makitensor_names = makitensor_info[MakiTensor.PARENT_TENSOR_NAMES]

            # If only one MakiTensor was used to create the current one,
            # then the layer does not expect a list as input
            if len(parent_makitensors) == 1:
                parent_makitensors = parent_makitensors[0]

            # Get the parent layer object and pass the parent makitensors through it.
            parent_layer = get_parent_layer(makitensor_info[MakiTensor.PARENT_LAYER_INFO])
            output_makitensors = parent_layer(parent_makitensors)

            # The rest of the code expects `output_makitensors` to be a list.
            if not isinstance(output_makitensors, list):
                output_makitensors = [output_makitensors]

            # Save the output makitensors to the dictionary.
            output_makitensors_names = parent_layer.get_children(parent_makitensor_names[0])
            for output_makitensor, output_makitensor_name in zip(output_makitensors, output_makitensors_names):
                makitensors[output_makitensor_name] = output_makitensor

            return makitensors[makitensor_name]

        def get_parent_layer(parent_layer_info, layer=None):
            """
            Builds the layer, saves to the `layers` dictionary and returns it or returns an already built layer.
            Parameters
            ----------
            parent_layer_info : dict
                Information for building the layer.
            layer : MakiLayer
                Already built layer object. This parameter is used only in cases
                when the input layer is supplied.
            Returns
            -------
            MakiLayer
                Built layer object.
            """
            params = parent_layer_info[MakiRestorable.PARAMS]
            name = params[MakiRestorable.NAME]

            # This IF statement only for cases when a custom InputLayer is passed.
            if layer is not None:
                # If a custom InputLayer is passed, it probably has a different name.
                # Therefore, we need to change 'parent_tensor_names' list for each makitensor there is in the graph.
                # P.S. If we don't do that we'll get an exception during graph restoration.
                old_name = name
                new_name = layer.get_name()
                for makitensor_name in graph_info:
                    mt_info = graph_info[makitensor_name]
                    # Change tensor names
                    new_parent_tensor_names = []
                    for parent_t_name in mt_info[MakiTensor.PARENT_TENSOR_NAMES]:
                        if parent_t_name == old_name:
                            # Found the old InputLayer, change its name to the new one
                            new_parent_tensor_names += [new_name]
                            continue
                        new_parent_tensor_names += [parent_t_name]
                    # Update parent tensor names
                    mt_info[MakiTensor.PARENT_TENSOR_NAMES] = new_parent_tensor_names
                # The input layer is a MakiTensor as well.
                makitensors[new_name] = layer

                layers[new_name] = layer
                name = new_name

            if layers.get(name) is None:
                layers[name] = MakiBuilder.__layer_from_dict(parent_layer_info)

            return layers[name]

        for name_output in outputs:
            restore_makitensor(name_output)

        return makitensors

    @staticmethod
    @abstractmethod
    def from_json(path_to_model):
        """
        Restore certain model.
        This method must be implemented by other models.
        """
        pass
