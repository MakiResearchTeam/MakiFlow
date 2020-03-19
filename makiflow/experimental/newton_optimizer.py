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
