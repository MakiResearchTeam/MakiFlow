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


class InitFabric(type):
    __registry = {}

    def __new__(mcs, class_name, parents, attributes):
        init_class = super(InitFabric, mcs).__new__(mcs, class_name, parents, attributes)
        InitFabric.__registry[class_name] = init_class
        return init_class

    @classmethod
    def get_init(mcs, name):
        return mcs.__registry.get(name)

