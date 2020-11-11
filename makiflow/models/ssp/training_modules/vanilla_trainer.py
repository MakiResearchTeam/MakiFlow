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
    @staticmethod
    def coordinates_name(feature_map_size):
        w, h = feature_map_size
        return f'COORDINATES_W={w}_H={h}'

    @staticmethod
    def point_visibility_indicators_name(feature_map_size):
        w, h = feature_map_size
        return f'POINT_VISIBILITY_INDICATORS_W={w}_H={h}'

    @staticmethod
    def human_presence_indicators_name(feature_map_size):
        w, h = feature_map_size
        return f'HUMAN_PRESENCE_INDICATORS_W={w}_H={h}'

    def _setup_label_placeholders(self):
        raise NotImplementedError('This method is not implemented for the SSP trainer. You need to pass'
                                  'in the necessary placeholders/Tensors yourself.')

    def _build_loss(self):
        pass

    def get_label_feed_dict_config(self):
        pass
