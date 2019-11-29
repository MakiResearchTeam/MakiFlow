import tensorflow as tf


class NewtonOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.grads = []
        self.hesses = []

    def variables(self):
        return self.grads + self.hesses

    def _compute_update(self, var, objective):
        grad = tf.squeeze(tf.gradients(ys=objective, xs=[var]), axis=0)
        hess = tf.squeeze(tf.hessians(ys=objective, xs=[var]))
        # self.grads += [grad]
        # self.hesses += [hess]
        return tf.matmul(tf.matrix_inverse(hess), grad)  # update

    def _apply_update(self, var, update):
        adjusted_update = self.learning_rate * update
        return tf.assign_sub(var, adjusted_update)

    def minimize(self, objective, train_vars=None):
        if train_vars is None:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = []
        for var in train_vars:
            update = self._compute_update(var, objective)
            update_ops += [self._apply_update(var, update)]
        return update_ops
