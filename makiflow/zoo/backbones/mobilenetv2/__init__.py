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

from .models import MobileNetV2_1_4, MobileNetV2_1_3, MobileNetV2_1_0, MobileNetV2_0_75
from .blocks import MobileNetV2InvertedResBlock
from .builder import build_MobileNetV2
from .utils import get_batchnorm_params, make_divisible

del absolute_import

