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
        outputs
        graph_info_json : dict
            Graph info section from the architecture file.
        input_layer : InputMakiLayer
            Custom InputLayer. Use this parameter if you want to train the model with pipelines
            or simply want to change the batch size.
        """
        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in graph_info_json:
            graph_info[tensor[MakiRestorable.NAME]] = tensor

        # Collects all the created MakiTensors.
        # Contains pairs {makitensor_name: MakiTensor}.
        makitensors = {}
        # Collects all the created layers.
        # Contains pairs {layer_name: MakiLayer}.
        layers = {}

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

            if layer is not None:
                layers[name] = layer

            if layers.get(name) is None:
                layers[name] = MakiBuilder.__layer_from_dict(parent_layer_info)

            return layers[name]

        def restore_in_and_out_x(makitensor_name):
            """
            Restore inputs and outputs of model from json.
            """
            # from_ - name of layer
            #
            makitensor_info = graph_info[makitensor_name]

            # Check if the makitensor was already created.
            if makitensors.get(makitensor_name) is not None:
                return makitensors[makitensor_name]

            parent_makitensor_names = makitensor_info[MakiTensor.PARENT_TENSOR_NAMES]
            # Check if we at the beginning of the graph. In this case we create InputLayer and return it.
            if len(parent_makitensor_names) == 0:
                layer = get_parent_layer(makitensor_info[MakiTensor.PARENT_LAYER_INFO], layer=input_layer)
                # The input layer is a MakiTensor as well.
                makitensors[makitensor_name] = layer
                return layer

            parent_makitensors = []
            for parent_makitensor_name in parent_makitensor_names:
                parent_makitensors += [restore_in_and_out_x(parent_makitensor_name)]

            print(parent_makitensors)
            # If only one MakiTensor was used to create the current one,
            # then the layer does not expect a list as input
            if len(parent_makitensors) == 1:
                parent_makitensors = parent_makitensors[0]

            # Get the parent layer object.
            parent_layer = get_parent_layer(makitensor_info[MakiTensor.PARENT_LAYER_INFO])
            # Get the output tensors of the parent layer and save them to the `makitensors` dictionary.
            output_makitensors = parent_layer(parent_makitensors)

            # The rest of the code expects `output_makitensors` to be a list.
            if not isinstance(output_makitensors, list):
                output_makitensors = [output_makitensors]

            output_makitensors_names = parent_layer.get_children(parent_makitensor_names[0])
            for output_makitensor, output_makitensor_name in zip(output_makitensors, output_makitensors_names):
                makitensors[output_makitensor_name] = output_makitensor

            return makitensors[makitensor_name]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return makitensors

    @staticmethod
    @abstractmethod
    def from_json(path_to_model):
        """
        Restore certain model.
        This method must be implemented by other models.
        """
        pass
