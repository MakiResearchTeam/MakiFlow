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
    def restore_graph(outputs, graph_info_json, batch_size, input_layer: InputMakiLayer = None):
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
        batch_size : int
            Soon.
        """
        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in graph_info_json:
            graph_info[tensor[MakiRestorable.NAME]] = tensor

        used = {}
        coll_tensors = {}

        def restore_in_and_out_x(from_):
            """
            Restore inputs and outputs of model from json.
            """
            # from_ - name of layer
            parent_layer_info = graph_info[from_]
            if used.get(from_) is None:
                used[from_] = True
                # like "to"
                all_parent_names = parent_layer_info[MakiTensor.PARENT_TENSOR_NAMES]
                # store ready tensors
                takes = []
                if len(all_parent_names) != 0:
                    # All layer except input layer
                    layer = MakiBuilder.__layer_from_dict(parent_layer_info[MakiTensor.PARENT_LAYER_INFO])
                    for elem in all_parent_names:
                        takes += [restore_in_and_out_x(elem)]
                    answer = layer(takes[0] if len(takes) == 1 else takes)
                else:
                    # Input layer
                    if input_layer is not None:
                        answer = input_layer
                    else:
                        temp = {}
                        temp.update({
                            MakiRestorable.FIELD_TYPE: parent_layer_info[MakiRestorable.FIELD_TYPE],
                            MakiRestorable.PARAMS: parent_layer_info[MakiRestorable.PARAMS]}
                        )
                        if batch_size is not None:
                            temp[MakiRestorable.PARAMS][InputMakiLayer.INPUT_SHAPE][0] = batch_size
                        answer = MakiBuilder.__layer_from_dict(temp)

                coll_tensors[from_] = answer
                return answer
            else:
                return coll_tensors[from_]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return coll_tensors

    @staticmethod
    @abstractmethod
    def from_json(path_to_model):
        """
        Restore certain model.
        This method must be implemented by other models.
        """
        pass
