from .tensorboard import TensorBoard
from makiflow.core.inference.maki_model import MakiModel
import tensorflow as tf


class Hermes(TensorBoard):
    def __init__(self, model: MakiModel):
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
