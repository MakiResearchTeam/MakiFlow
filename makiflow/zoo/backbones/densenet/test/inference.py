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
from makiflow.zoo.backbones.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264


class DenseNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: DenseNet121(x, create_model=True),
            lambda x: DenseNet161(x, create_model=True),
            lambda x: DenseNet169(x, create_model=True),
            lambda x: DenseNet201(x, create_model=True),
            lambda x: DenseNet264(x, create_model=True)
        ]


