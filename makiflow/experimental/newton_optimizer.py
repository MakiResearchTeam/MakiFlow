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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops


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

    NEWTON_MODE = 0
    SGD_MODE = 0

    def __init__(self, learning_rate, learning_rate_sgd=1e-4, name='NewtonOptimizer'):
        self._name = name
        self._lr = learning_rate
        self._lr_sgd = learning_rate_sgd
        self._current_state = None

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._lr_sgd_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name='learning_rate')
        self._lr_sgd_t = ops.convert_to_tensor(self._lr_sgd, name='learning_rate_sgd')

    def variables(self):
        return [self._lr_t, self._lr_sgd_t]

    def _compute_update(self, var, objective):
        grad = tf.squeeze(tf.gradients(ys=objective, xs=var), axis=0, name=self._name + '_squeeze_grads')
        #hess = tf.squeeze(tf.hessians(ys=objective, xs=[var]), name=self._name + '_squeeze_hess')

        hess = tf.transpose(jacobian(objective, var))
        # Check rank of the matrix
        matrix_rank = tf.linalg.matrix_rank(a=hess, name=self._name + '_matrix_rank_of_hesse_result')

        # Number of row must be equal with matrix rank

        self._current_state = tf.cond(tf.math.equal(matrix_rank, hess.get_shape()[0]),
                        lambda: NewtonOptimizer.NEWTON_MODE,                                        # newton mode
                        lambda: NewtonOptimizer.SGD_MODE                                            # sgd mode
        )

        flat_grad = tf.reshape(grad, shape=[1, -1])
        newton_update = tf.matmul(flat_grad, tf.matrix_inverse(hess),
                         name=self._name + '_matmul_gradient_and_inverse_hesse'
        )

        newton_update = tf.reshape(newton_update, grad.get_shape())

        update = tf.cond(tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE),
                        lambda: newton_update,                                                                          # newton mode
                        lambda: grad                                                                # sgd mode
        )


        # self.grads += [grad]
        # self.hesses += [hess]

        return update

    def _apply_update(self, var, update):
        adjusted_update = tf.cond(tf.math.equal(self._current_state, NewtonOptimizer.NEWTON_MODE),
                            lambda: self._lr_t * update,                                             # newton mode
                            lambda: self._lr_sgd_t * update                                          # sgd mode
        )
        return tf.assign_sub(var, adjusted_update)

    def minimize(self, objective, var_list=None):
        self._prepare()

        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = []
        for var in var_list:
            update = self._compute_update(var, objective)
            update_ops += [self._apply_update(var, update)]
        return update_ops
