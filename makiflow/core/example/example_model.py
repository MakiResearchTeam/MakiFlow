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

from makiflow.core import MakiModel, MakiTensor


class ExampleModel(MakiModel):

    def get_feed_dict_config(self) -> dict:
        """
        This method will be used by trainers for feeding data into the model.
        """
        return {
            self._in_x: 0
        }

    def _get_model_info(self):
        return {
            'param': 'value'
        }

    def __init__(self, in_x: MakiTensor, out_x: MakiTensor, name='ExampleModel'):
        super().__init__(
            outputs=[out_x],
            inputs=[in_x]
        )
        self._name = name
        self._out_x = out_x
        self._in_x = in_x

    def predict(self, data):
        return self.get_session().run(
            self._out_x.get_data_tensor(),
            feed_dict={
                self._in_x.get_data_tensor(): data
            }
        )

    def get_output(self):
        return self._out_x
