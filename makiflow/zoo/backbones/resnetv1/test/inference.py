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
from makiflow.zoo.backbones.resnetv1 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
    Little_ResNet20, Little_ResNet110, Little_ResNet32, Little_ResNet44, Little_ResNet56


class DenseNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: ResNet18(x, create_model=True),
            lambda x: ResNet34(x, create_model=True),
            lambda x: ResNet50(x, create_model=True),
            lambda x: ResNet101(x, create_model=True),
            lambda x: ResNet152(x, create_model=True),
            lambda x: Little_ResNet20(x, create_model=True),
            lambda x: Little_ResNet110(x, create_model=True),
            lambda x: Little_ResNet32(x, create_model=True),
            lambda x: Little_ResNet44(x, create_model=True),
            lambda x: Little_ResNet56(x, create_model=True),
        ]


