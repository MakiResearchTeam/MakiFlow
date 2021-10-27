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

from makiflow.zoo.testing import InferenceTest
from makiflow.zoo.backbones.mobilenetv2 import MobileNetV2_0_75, MobileNetV2_1_0, MobileNetV2_1_3, MobileNetV2_1_4


class MobileNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: MobileNetV2_0_75(x, create_model=True),
            lambda x: MobileNetV2_1_0(x, create_model=True),
            lambda x: MobileNetV2_1_3(x, create_model=True),
            lambda x: MobileNetV2_1_4(x, create_model=True),
        ]


