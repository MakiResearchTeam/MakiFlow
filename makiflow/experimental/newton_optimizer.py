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
import numpy as np


class NewtonOptimizer:
    """
    Implementation of Newton Optimizer.
    See https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    """
    NEWTON_LEARNING_RATE = 'NEWTON_LEARNING_RATE'
    SGD_LEARNING_RATE = 'SGD_LEARNING_RATE'

    HESSIAN_RANK = 'HESSIAN_RANK'
    GLOBAL_NEWTON_UPDATE_VECTOR = 'GLOBAL_NEWTON_UPDATE_VECTOR'

    NEWTON_UPDATE = 'NEWTON_UPDATE'
    SGD_UPDATE = 'SGD_UPDATE'

    def __init__(self, learning_rate, learning_rate_sgd=1e-4, alpha=0.1, name='NewtonOptimizer'):
        self._name = name
        self._alpha = alpha

        with tf.name_scope(name):
            self._lr = tf.convert_to_tensor(learning_rate, name=NewtonOptimizer.NEWTON_LEARNING_RATE)
            self._lr_sgd = tf.convert_to_tensor(learning_rate_sgd, name=NewtonOptimizer.SGD_LEARNING_RATE)

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
        self._flattened_params_shape = [
            int(param.get_shape().as_list()[0]) for param in self._flattened_params
        ]
        self._grad_vector = self._flatten(self._grads)

        self._total_params_elements = 0
        for param_len in self._flattened_params_shape:
            self._total_params_elements += param_len

    def _flatten(self, tensors):
        return tf.concat([tf.reshape(tensor, [-1]) \
                          for tensor in tensors], axis=0)

    def _get_hess_v_op(self, v):
        # Flatten the gradient
        vprod = tf.math.multiply(self._grad_vector, tf.stop_gradient(v))
        Hv_op = self._flatten(tf.gradients(vprod, self._params))
        return Hv_op

    def _compute_hessian(self):
        """
        grad : tf.Tensor
            Computed gradient of a function with respect to `param`.
        """
        self._hessian = tf.map_fn(
            self._get_hess_v_op,
            tf.eye(self._total_params_elements, self._total_params_elements),
            dtype='float32'
        ) + self._alpha * tf.eye(self._total_params_elements, self._total_params_elements)

    # noinspection PyMethodMayBeStatic
    def variables(self):
        return []

    def _compute_newton_var_updates(self):
        # Compute global update vector
        grad_vector = tf.reshape(self._grad_vector, shape=[1, -1])
        newton_update = tf.matmul(
            grad_vector,
            tf.matrix_inverse(self._hessian),
            name=NewtonOptimizer.GLOBAL_NEWTON_UPDATE_VECTOR
        )

        # Factorize the global update vector into its initial components
        start_ind = 0

        var_updates = []
        for flat_len, init_shape, param in zip(self._flattened_params_shape, self._params_shape, self._params):
            flat_update = newton_update[0, start_ind: start_ind + flat_len]
            restored_update = tf.reshape(flat_update, init_shape)
            if param.constraint is None:
                var_updates += [tf.assign(param, param - restored_update * self._lr)]
            else:
                var_updates += [tf.assign(param, param.constraint(param - restored_update * self._lr))]
            start_ind += flat_len

        return tf.group(*var_updates, name=NewtonOptimizer.NEWTON_UPDATE)

    def _compute_grad_var_updates(self):
        var_updates = []
        for param, grad in zip(self._params, self._grads):
            var_updates += [tf.assign_sub(param, grad * self._lr_sgd)]

        return tf.group(*var_updates, name=NewtonOptimizer.SGD_UPDATE)

    def _compute_update_op(self):
        with tf.name_scope(self._name):
            matrix_rank = tf.linalg.matrix_rank(
                a=self._hessian,
                tol=0.00001,
                name=NewtonOptimizer.HESSIAN_RANK
            )

            # Number of row must be equal with matrix rank
            condition = tf.math.equal(matrix_rank, self._hessian.get_shape()[0])
            self._update_op = tf.cond(
                condition,
                self._compute_newton_var_updates,  # newton mode
                self._compute_grad_var_updates  # sgd mode
            )

    def minimize(self, objective, var_list=None, global_step=None):
        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self._init_hessian_vars(var_list, objective)
        self._compute_hessian()
        self._compute_update_op()

        return self._update_op
