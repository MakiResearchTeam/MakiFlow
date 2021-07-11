import tensorflow as tf

from .weights_regularization import WeightRegularization, Model


class L2(WeightRegularization):
    def __init__(self, model: Model, decay=1e-6):
        """
        Applies l2 regularization to the model's weights.

        Parameters
        ----------
        model : Model
            The model.
        decay : float or dict
            If decay is float, regularization will be applied to all weights.
            Otherwise decay must be a dictionary containing pairs (layer_name : decay_val).
        """
        reg_fn = lambda t: tf.nn.l2_loss(t)
        super().__init__(model=model, reg_fn=reg_fn, decay=decay)
