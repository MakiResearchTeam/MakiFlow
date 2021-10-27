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


from .models import (ResNet18, ResNet34, ResNet50,
                     ResNet101, ResNet152, Little_ResNet20,
                     Little_ResNet32, Little_ResNet44,
                     Little_ResNet56, Little_ResNet110)

from .blocks import (ResNetIdentityBlockV1, ResNetConvBlockV1,
                     ResNetConvBlock_woPointWiseV1, ResNetIdentityBlock_woPointWiseV1)

from .builder import build_ResNetV1, build_LittleResNetV1
from .utils import get_batchnorm_params
