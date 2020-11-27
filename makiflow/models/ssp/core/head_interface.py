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

from abc import ABC, abstractmethod


class HeadInterface(ABC):
    @abstractmethod
    def get_bbox_configuration(self) -> list:
        pass

    @abstractmethod
    def get_coords(self):
        pass

    @abstractmethod
    def get_point_indicators(self):
        pass

    @abstractmethod
    def get_human_indicators(self):
        pass

    @abstractmethod
    def get_grid_size(self) -> list:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass
