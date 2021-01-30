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

from .train_graph_compiler import TrainGraphCompiler


class Serializer(TrainGraphCompiler):
    TYPE = 'type'
    PARAMS = 'params'

    def to_dict(self):
        assert self.TYPE != Serializer.TYPE, 'The trainer did not specified its type via static TYPE variable.'
        return {
            Serializer.TYPE: self.TYPE,
            Serializer.PARAMS: {}
        }

    def set_params(self, params):
        """
        This method must be overloaded if the trainer uses some additional
        parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the trainer.
        """
        pass



