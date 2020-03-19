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

from abc import abstractmethod


class MapMethod:
    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass


class PostMapMethod(MapMethod):
    def __init__(self):
        self._parent_method = None

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass

    def __call__(self, parent_method: MapMethod):
        self._parent_method = parent_method
        return self
