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

from .training_modules import FocalTrainingModule, MakiTrainingModule, QuadraticCrossEntropyTrainingModule, \
    WeightedFocalTrainingModule, WeightedCrossEntropyTrainingModule

import json
from makiflow.core.maki_entities import MakiCore
from .main_modules import SegmentatorBasic

class Segmentator(
    FocalTrainingModule,
    MakiTrainingModule,
    QuadraticCrossEntropyTrainingModule,
    WeightedCrossEntropyTrainingModule,
    WeightedFocalTrainingModule
):

    @staticmethod
    def from_json(path_to_model, input_tensor=None):
        json_file = open(path_to_model)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info[MakiCore.MODEL_INFO][SegmentatorBasic.OUTPUT_MT]
        input_tensor_name = json_info[MakiCore.MODEL_INFO][SegmentatorBasic.INPUT_MT]
        model_name = json_info[MakiCore.MODEL_INFO][SegmentatorBasic.NAME]

        graph_info = json_info[MakiCore.GRAPH_INFO]

        inputs_outputs = MakiCore.restore_graph(
            [output_tensor_name], graph_info, input_layer=input_tensor
        )
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        model = Segmentator(input_s=in_x, output=out_x, name=model_name)

        print('Model is restored!')
        return model
