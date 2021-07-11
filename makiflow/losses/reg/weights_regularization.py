from makiflow.core import Loss, Model


class WeightRegularization(Loss):
    def __init__(self, model: Model, reg_fn, decay=1e-6):
        """
        A class for generic weights regularization.

        Parameters
        ----------
        model : Model
            The model.
        reg_fn : function
            A function with signature (param: tf.Variable) that computes regularization loss scalar for
            the given parameter.
        decay : float or dict
            If decay is float, regularization will be applied to all weights.
            Otherwise decay must be a dictionary containing pairs (layer_name : decay_val).
        """
        super().__init__([], {}, None)
        self._model = model
        self._reg_fn = reg_fn
        if not isinstance(decay, dict):
            decay_upd = {}
            layers = model.layers
            for layer_name in layers.keys():
                decay_upd[layer_name] = decay
            decay = decay_upd
        self._decay = decay

    @property
    def model(self):
        return self._model

    @property
    def decay(self):
        return self._decay

    def build(self, tensor_provider):
        if self.loss is not None:
            return self.loss

        loss = 0.0
        for layer_name, decay in self.decay.items():
            for param in self.model.get_layer(layer_name).get_params_regularize():
                loss += self._reg_fn(param) * decay

        self._loss = loss
        return self.loss
