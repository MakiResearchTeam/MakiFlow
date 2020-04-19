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

import tensorflow as tf
# This class defines the API to add Ops to train a model.
from tensorflow.python.framework import ops


class NewtonOptimizer:
    """
    Implementation of Newton Optimizer.
    See https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    """
    NEWTON_LEARNING_RATE = 'NEWTON_LEARNING_RATE'
    SGD_LEARNING_RATE = 'SGD_LEARNING_RATE'
    NEWTON_MODE = tf.constant(0, dtype=tf.int32, name="NEWTON_MODE")
    SGD_MODE = tf.constant(1, dtype=tf.int32, name='SGD_MODE')
    HESSIAN_RANK = 'HESSIAN_RANK'
    NEWTON_UPDATE = 'NEWTON_UPDATE'

    def __init__(self, learning_rate, learning_rate_sgd=1e-4, name='NewtonOptimizer'):
        self._name = name
        self._current_state = None

        with tf.name_scope(name):
            self._lr = ops.convert_to_tensor(learning_rate, name=NewtonOptimizer.NEWTON_LEARNING_RATE)
            self._lr_sgd = ops.convert_to_tensor(learning_rate_sgd, name=NewtonOptimizer.SGD_LEARNING_RATE)

            self._prepare_hessian_vars()

    def _prepare_hessian_vars(self):
        self._params_shape = None
        self._flattened_params = None
        self._flattened_params_shape = None

        self._params_vector = None
        self._grad_vector = None

    def _init_hessian_vars(self, params, objective):
        self._params = params
        self._grads = tf.gradients(objective, params)
        self._params_shape = [param.get_shape() for param in params]
        self._flattened_params = [tf.reshape(param, [-1]) for param in params]
        self._flattened_params_shape = [param.get_shape() for param in self._flattened_params]

        self._params_vector = tf.concat(self._flattened_params_shape, axis=0)
        self._grad_vector = tf.concat(
            [tf.reshape(grad, [-1]) for grad in self._grads]
        )

    def _get_hess_v_op(self, v):
        # Flatten the gradient
        vprod = tf.math.multiply(self._grad_vector, tf.stop_gradient(v))
        Hv_op = tf.reshape(tf.gradients(vprod, self._params_vector), [-1])
        return Hv_op

    def _compute_hessian(self):
        """
        grad : tf.Tensor
            Computed gradient of a function with respect to `param`.
        """
        n_params = self._params_vector.get_shape().as_list()[0]
        self._hessian = tf.map_fn(
            self._get_hess_v_op,
            tf.eye(n_params, n_params),
            dtype='float32'
        )

    # noinspection PyMethodMayBeStatic
    def variables(self):
        return []

    def _compute_newton_updates(self):
        # Compute global update vector
        newton_update = tf.matmul(self._grad_vector, tf.matrix_inverse(self._hessian))

        # Factorize the global update vector into its initial components
        updates = []
        start_ind = 0
        for flat_len, init_shape in zip(self._flattened_params_shape, self._params_shape):
            flat_update = newton_update[start_ind: start_ind + flat_len]
            updates += [tf.reshape(flat_update, init_shape)]
        return updates

    def _compute_updates(self):
        with tf.name_scope(self._name):
            matrix_rank = tf.linalg.matrix_rank(
                a=self._hessian,
                name=NewtonOptimizer.HESSIAN_RANK
            )

            # Number of row must be equal with matrix rank
            condition = tf.math.equal(matrix_rank, self._hessian.get_shape()[0])
            self._current_state = tf.where(
                condition,
                NewtonOptimizer.NEWTON_MODE,
                NewtonOptimizer.SGD_MODE
            )

            def compute_grad_updates():
                return self._grads

            condition = tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE)
            self._updates = tf.cond(
                condition,
                self._compute_newton_updates,   # newton mode
                compute_grad_updates            # sgd mode
            )

    def _apply_update(self, var, update):
        adjusted_update = tf.where(
            tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE),
            self._lr * update,  # newton mode
            self._lr_sgd * update  # sgd mode
        )
        return tf.assign_add(var, adjusted_update)

    def minimize(self, objective, var_list=None):
        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self._init_hessian_vars(var_list, objective)
        self._compute_hessian()
        self._compute_updates()

        update_ops = []
        for param, update in zip(self._params, self._updates):
            update_ops += [self._apply_update(param, update)]

        return tf.group(*update_ops, name=NewtonOptimizer.NEWTON_UPDATE)
