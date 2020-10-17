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

from __future__ import absolute_import
import json

from makiflow.models.classificator import Classificator
from makiflow.models.classificator.main_modules import CParams

from makiflow.layers.trainable_layers import TrainableLayerAddress
from makiflow.layers.untrainable_layers import UnTrainableLayerAddress
from makiflow.layers.rnn_layers import RNNLayerAddress

from makiflow.core.graph_entities.maki_layer import MakiRestorable
from makiflow.core.graph_entities.maki_tensor import MakiTensor
from makiflow.core.inference.maki_model import MakiModel

from makiflow.models.ssd.detector_classifier import DetectorClassifier, DCParams
from makiflow.models import SSDModel
from makiflow.models import Segmentator
from makiflow.models import TextRecognizer

class Builder:

    @staticmethod
    def classificator_from_json(json_path):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info[MakiModel.MODEL_INFO][CParams.OUTPUT_MT]
        input_tensor_name = json_info[MakiModel.MODEL_INFO][CParams.INPUT_MT]
        model_name = json_info[MakiModel.MODEL_INFO][CParams.NAME]

        graph_info = json_info[MakiModel.GRAPH_INFO]

        inputs_outputs = Builder.restore_graph([output_tensor_name], graph_info)
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        print('Model is restored!')
        return Classificator(input=in_x, output=out_x, name=model_name)

    @staticmethod
    def ssd_from_json(json_path, generator=None):
        """Creates and returns SSDModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict[MakiModel.MODEL_INFO]['name']
        # Collect names of the MakiTensors that are inputs for the DetectorClassifiers
        # for restoring the graph.
        dcs_dicts = architecture_dict[MakiModel.MODEL_INFO]['dcs']
        outputs = []
        for dcs_dict in dcs_dicts:
            params = dcs_dict['params']
            outputs += [params['reg_x_name'], params['class_x_name']]

        graph_info = architecture_dict[MakiModel.GRAPH_INFO]
        inputs_outputs = Builder.restore_graph(outputs, graph_info, generator)
        # Restore all the DetectorClassifiers
        dcs = []
        for dc_dict in architecture_dict[MakiModel.MODEL_INFO]['dcs']:
            dcs.append(Builder.__detector_classifier_from_dict(dc_dict, inputs_outputs))
        input_name = architecture_dict[MakiModel.MODEL_INFO]['input_s']
        input_s = inputs_outputs[input_name]
        offset_reg_type = architecture_dict[MakiModel.MODEL_INFO]['reg_type']
        print('Model is recovered.')

        return SSDModel(dcs=dcs, input_s=input_s, name=name, offset_reg_type=offset_reg_type)

    @staticmethod
    def __detector_classifier_from_dict(dc_dict, inputs_outputs):
        """Creates and returns DetectorClassifier from dictionary"""
        params = dc_dict[MakiRestorable.PARAMS]
        name = params[DCParams.NAME]
        class_number = params[DCParams.CLASS_NUMBER]
        dboxes = params[DCParams.DBOXES]

        rkw = params[DCParams.RKW]
        rkh = params[DCParams.RKH]
        rin_f = params[DCParams.RIN_F]
        use_reg_bias = params[DCParams.USE_REG_BIAS]
        reg_init_type = params[DCParams.REG_INIT_TYPE]

        ckw = params[DCParams.CKW]
        ckh = params[DCParams.CKH]
        cin_f = params[DCParams.CIN_F]
        use_class_bias = params[DCParams.USE_CLASS_BIAS]
        class_init_type = params[DCParams.CLASS_INIT_TYPE]

        reg_x = inputs_outputs[params[DCParams.REG_X_NAME]]
        class_x = inputs_outputs[params[DCParams.CLASS_X_NAME]]

        return DetectorClassifier(
            reg_fms=reg_x, rkw=rkw, rkh=rkh, rin_f=rin_f,
            class_fms=class_x, ckw=ckw, ckh=ckh, cin_f=cin_f,
            num_classes=class_number, dboxes=dboxes, name=name,
            use_class_bias=use_class_bias, use_reg_bias=use_reg_bias,
            reg_init_type=reg_init_type, class_init_type=class_init_type
        )

    @staticmethod
    def text_recognizer_from_json(json_path):
        """Creates and returns TextRecognizer from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        chars = architecture_dict['chars']
        max_seq_length = architecture_dict['max_seq_length']
        decoder_type = architecture_dict['decoder_type']

        cnn_layers = []
        for layer in architecture_dict['cnn_layers']:
            cnn_layers.append(Builder.__layer_from_dict(layer))
        rnn_layers = []
        for layer in architecture_dict['rnn_layers']:
            rnn_layers.append(Builder.__layer_from_dict(layer))

        return TextRecognizer(
            cnn_layers=cnn_layers,
            rnn_layers=rnn_layers,
            input_shape=input_shape,
            chars=chars,
            max_seq_length=max_seq_length,
            decoder_type=decoder_type,
            name=name
        )

    @staticmethod
    def segmentator_from_json(json_path, generator=None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info[MakiModel.MODEL_INFO]['output']
        input_tensor_name = json_info[MakiModel.MODEL_INFO]['input_s']
        model_name = json_info[MakiModel.MODEL_INFO]['name']

        MakiTensors_of_model = json_info[MakiModel.GRAPH_INFO]

        inputs_outputs = Builder.restore_graph(
            [output_tensor_name], MakiTensors_of_model, generator
        )
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        model = Segmentator(input_s=in_x, output=out_x, name=model_name)
        if generator is not None:
            model.set_generator(generator)
        print('Model is restored!')
        return model

# -----------------------------------------------------------LAYERS RESTORATION-----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __layer_from_dict(layer_dict):
        """
        Creates and returns Layer from dictionary
        """

        # Collects the address to all existing layers
        all_layers_adress = {}
        all_layers_adress.update(RNNLayerAddress.ADDRESS_TO_CLASSES)
        all_layers_adress.update(TrainableLayerAddress.ADDRESS_TO_CLASSES)
        all_layers_adress.update(UnTrainableLayerAddress.ADDRESS_TO_CLASSES)

        params = layer_dict[MakiRestorable.PARAMS]

        build_layer = all_layers_adress.get(layer_dict[MakiRestorable.FIELD_TYPE])

        if build_layer is None:
            raise KeyError(f'{layer_dict[MakiRestorable.FIELD_TYPE]} was not found!')

        return build_layer.build(params)

    # -----------------------------------------------------------GRAPH RESTORATION--------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def restore_graph(outputs, graph_info_json, generator=None):
        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in graph_info_json:
            graph_info[tensor[MakiRestorable.NAME]] = tensor

        used = {}
        coll_tensors = {}

        def restore_in_and_out_x(from_):
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
                    layer = Builder.__layer_from_dict(parent_layer_info[MakiTensor.PARENT_LAYER_INFO])
                    for elem in all_parent_names:
                        takes += [restore_in_and_out_x(elem)]
                    answer = layer(takes[0] if len(takes) == 1 else takes)
                else:
                    # Input layer
                    if generator is not None:
                        answer = generator
                    else:
                        temp = {}
                        temp.update({
                            MakiRestorable.FIELD_TYPE: parent_layer_info[MakiRestorable.FIELD_TYPE],
                            MakiRestorable.PARAMS: parent_layer_info[MakiRestorable.PARAMS]}
                        )
                        answer = Builder.__layer_from_dict(temp)

                coll_tensors[from_] = answer
                return answer
            else:
                return coll_tensors[from_]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return coll_tensors
