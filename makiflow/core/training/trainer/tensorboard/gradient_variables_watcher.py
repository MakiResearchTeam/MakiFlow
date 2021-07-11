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

from .tensorboard import TensorBoard
from makiflow.core.inference.maki_core import MakiCore
import tensorflow as tf


class GradientVariablesWatcher(TensorBoard):
    def __init__(self, model: MakiCore):
        super().__init__()
        self._model = model
        self._layers_to_show = []
        self._var2grad = None

    def set_vars_grads(self, var_grad):
        self._var2grad = dict(var_grad)

    def get_var2grad(self):
        if self._var2grad is None:
            return None
        return self._var2grad.copy()

    def set_layers_histograms(self, layer_names):
        self._layers_to_show = layer_names

    def setup_tensorboard(self):
        # Collect all layers histograms
        for layer_name in self._layers_to_show:
            # Add weights histograms
            layer_weights = self._model.get_layer(layer_name).get_params()
            with tf.name_scope(f'{layer_name}/weight'):
                for weight in layer_weights:
                    self.add_histogram(weight, weight.name)

            # Add grads histograms
            with tf.name_scope(f'{layer_name}/grad'):
                for weight in layer_weights:
                    grad = self._var2grad.get(weight)
                    if grad is None:
                        print(f'Did not find gradient for layer={layer_name}, var={weight.name}')
                        continue
                    self.add_summary(tf.summary.histogram(name=weight.name, values=grad))
        super().setup_tensorboard()
