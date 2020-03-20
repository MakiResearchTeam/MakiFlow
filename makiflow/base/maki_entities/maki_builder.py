from abc import abstractmethod
from .maki_layer import MakiRestorable
from .maki_tensor import MakiTensor

from makiflow.layers.trainable_layers import TrainableLayerAddress
from makiflow.layers.untrainable_layers import UnTrainableLayerAddress, InputLayer
from makiflow.layers.rnn_layers import RNNLayerAddress


class MakiBuilder:
    # Provides API for model restoration.

    # Collects the addresses to all existing layers
    ALL_LAYERS_ADDRESS = {} \
        .update(RNNLayerAddress.ADDRESS_TO_CLASSES) \
        .update(TrainableLayerAddress.ADDRESS_TO_CLASSES) \
        .update(UnTrainableLayerAddress.ADDRESS_TO_CLASSES)

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
    def restore_graph(outputs, graph_info_json, batch_sz=None, generator=None):
        """
        Restore Inference graph with inputs and outputs of model from json.
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
                    temp = {}
                    temp.update({
                        MakiRestorable.FIELD_TYPE: parent_layer_info[MakiRestorable.FIELD_TYPE],
                        MakiRestorable.PARAMS: parent_layer_info[MakiRestorable.PARAMS]}
                    )
                    if batch_sz is not None:
                        temp[MakiRestorable.PARAMS][InputLayer.INPUT_SHAPE][0] = batch_sz
                    if generator is not None:
                        answer = generator
                    else:
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
