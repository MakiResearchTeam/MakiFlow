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


class HessianEstimator:
    @staticmethod
    def _get_Hv_op(v, grad, param):
        # Flatten the gradient
        grad = tf.reshape(grad, [-1])
        vprod = tf.math.multiply(grad, tf.stop_gradient(v))
        Hv_op = tf.reshape(tf.gradients(vprod, param), [-1])
        return Hv_op

    @staticmethod
    def compute_hessian(grad, param):
        """
        grad : tf.Tensor
            Computed gradient of a function with respect to `param`.
        """
        n_params = tf.reshape(param, shape=[-1]).get_shape().as_list()[0]
        H_op = tf.map_fn(
            lambda v: HessianEstimator._get_Hv_op(v, grad, param),
            tf.eye(n_params, n_params),
            dtype='float32'
        )
        return H_op


def jacobian(y, x, consider_batchsize=False):
    """
    Computes Jacobian of the function `y` given arguments `x`.

    Parameters
    ----------
    y : tf.Tensor
        A Tensor of arbitrary shape.

    x : tf.Tensor or tf.Variable
        A Tensor of arbitrary shape. `x` must be the original Tensor that was used for
        creation of the `y`. Do not perform any transformations on it before passing it.
    """
    y_flat = tf.reshape(y, [-1])  # [...] -> [lenght]
    length = y_flat.get_shape().as_list()[0]
    gradients = []
    for el_ind in range(length):
        # Computes one entry of the matrix
        gradients += [tf.reshape(tf.gradients(y_flat[el_ind], x), shape=[-1])]
    return tf.stack(gradients, axis=0)


class NewtonOptimizer():
    """
    Implementation of Newton Optimizer.
    See https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    """

    NEWTON_MODE = tf.constant(0.0, dtype=tf.float32, name="Newtone_mode")
    SGD_MODE = tf.constant(1.0, dtype=tf.float32, name='SGD_mode')

    def __init__(self, learning_rate, learning_rate_sgd=1e-4, name='NewtonOptimizer'):
        self._name = name
        self._lr = learning_rate
        self._lr_sgd = learning_rate_sgd
        self._current_state = None

        self._lr_t = ops.convert_to_tensor(self._lr, name='learning_rate')
        self._lr_sgd_t = ops.convert_to_tensor(self._lr_sgd, name='learning_rate_sgd')

    def variables(self):
        return []

    def _compute_update(self, var, objective):
        grad = tf.gradients(ys=objective, xs=var)[0]
        hess = HessianEstimator.compute_hessian(grad, var)
        matrix_rank = tf.linalg.matrix_rank(a=hess, name=self._name + '_matrix_rank_of_hesse_result')

        # Number of row must be equal with matrix rank

        self._current_state = tf.where(tf.math.equal(matrix_rank, hess.get_shape()[0]),
                                       NewtonOptimizer.NEWTON_MODE,  # newton mode
                                       NewtonOptimizer.SGD_MODE  # sgd mode
                                       )

        def compute_newton_update():
            flat_grad = tf.reshape(grad, shape=[1, -1])
            newton_update = tf.matmul(flat_grad, tf.matrix_inverse(hess))
            newton_update = tf.reshape(newton_update, grad.get_shape())
            return newton_update

        def compute_grad_update():
            return tf.multiply(grad, p)

        update = tf.cond(tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE),
                         compute_newton_update,  # newton mode
                         compute_grad_update  # sgd mode
                         )

        return update

    def _apply_update(self, var, update):
        adjusted_update = tf.where(tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE),
                                   self._lr_t * update,  # newton mode
                                   self._lr_sgd_t * update  # sgd mode
                                   )
        return tf.assign_add(var, adjusted_update)

    def minimize(self, objective, var_list=None):

        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = []
        for var in var_list:
            update = self._compute_update(var, objective)
            update_ops += [self._apply_update(var, update)]
        return update_ops
