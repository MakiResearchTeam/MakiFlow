# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiZoo.
#
# MakiZoo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiZoo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

MODEL_05 = '0.5'
MODEL_10 = '1.0'
MODEL_15 = '1.5'
MODEL_20 = '2.0'


SIZE_2_CONFIG_MODEL = {
    MODEL_05: [(48, 4), (96, 8), (192, 4), 1024],
    MODEL_10: [(116, 4), (232, 8), (464, 4), 1024],
    MODEL_15: [(176, 4), (352, 8), (704, 4), 1024],
    MODEL_20: [(244, 4), (488, 8), (976, 4), 2048],
}