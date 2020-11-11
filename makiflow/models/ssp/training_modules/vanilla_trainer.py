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

from makiflow.core import MakiBuilder, MakiTrainer


class VanillaTrainer(MakiTrainer):
    COORDINATES = 'COORDINATES'
    POINT_VISIBILITY_INDICATORS = 'POINT_VISIBILITY_INDICATORS'
    HUMAN_PRESENCE_INDICATORS = 'HUMAN_PRESENCE_INDICATORS'

    @staticmethod
    def encode(tensor_type, feature_map_size, bbox_config):
        h, w = feature_map_size
        h_scale, w_scale = bbox_config
        return f'{tensor_type}_WH={h}/{w}_BC={w_scale}/{h_scale}'

    @staticmethod
    def coordinates_name(feature_map_size, bbox_config):
        return VanillaTrainer.encode(VanillaTrainer.COORDINATES, feature_map_size, bbox_config)

    @staticmethod
    def point_visibility_indicators_name(feature_map_size, bbox_config):
        return VanillaTrainer.encode(VanillaTrainer.COORDINATES, feature_map_size, bbox_config)

    @staticmethod
    def human_presence_indicators_name(feature_map_size, bbox_config):
        return VanillaTrainer.encode(VanillaTrainer.COORDINATES, feature_map_size, bbox_config)

    @staticmethod
    def decode(tensor_name):
        tensor_type, feature_map_size, bbox_config = tensor_name.split('_')

        feature_map_size = feature_map_size.replace('WH=', '')
        h, w = feature_map_size.split('/')
        h, w = int(h), int(w)

        feature_map_size = feature_map_size.replace('BC=', '')
        w_scale, h_scale = feature_map_size.split('/')
        w_scale, h_scale = float(w_scale), float(h_scale)
        return tensor_type, (h, w), (w_scale, h_scale)

    def _setup_label_placeholders(self):
        raise NotImplementedError('This method is not implemented for the SSP trainer. You need to pass'
                                  'in the necessary placeholders/Tensors yourself.')

    def _init(self):
        super()._init()


    def _build_loss(self):
        pass

    def get_label_feed_dict_config(self):
        pass
