from .tensorboard import TensorBoard


class Hermes(TensorBoard):
    def __init__(self):
        super().__init__()
        self._layers_to_show = []
        self._var_grad = []

    def set_vars_grads(self, var_grad):
        self._var_grad = var_grad

